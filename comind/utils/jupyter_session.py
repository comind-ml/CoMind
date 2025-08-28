import tempfile
import atexit
import json
from time import time
from pathlib import Path
from dataclasses import dataclass
from jupyter_client import KernelManager, BlockingKernelClient
from jupyter_client.kernelspec import KernelSpecManager

from comind.config import Config, LLMConfig
from comind.utils.llm import query_llm

@dataclass
class ExecutionResult:
    success: bool
    llm_terminated: bool
    timeout: bool
    output: str
    error: str
    execution_time: float 

class JupyterSession:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        kernel_name = self._setup_uv_environment()

        self.km: KernelManager = KernelManager(kernel_name=kernel_name)
        self.km.start_kernel()
        self.kc: BlockingKernelClient = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready()
        self.log_file_path = self._get_log_file_path()

        atexit.register(self._safe_shutdown)
    
    def shutdown(self):
        self._safe_shutdown()

    def _setup_uv_environment(self) -> str:
        venv_py = self.cfg.agent_workspace_dir / ".venv" / "bin" / "python"
        assert venv_py.exists(), f"Virtual environment python not found at {venv_py}"
        kname = f"{self.cfg.competition_id}-{self.draft.id}"
        ksm = KernelSpecManager()
        
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "kernel.json").write_text(json.dumps({
                "argv": [str(venv_py), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": kname,
                "language": "python",
            }, indent=2), encoding="utf-8")
            ksm.install_kernel_spec(str(td), kernel_name=kname, user=True, replace=True)

        return kname

    def _safe_shutdown(self):
        self.kc.stop_channels()
        try:
            self.km.shutdown_kernel(now=False)
        except Exception:
            self.km.shutdown_kernel(now=True)
    
    def _get_log_file_path(self) -> Path:
        """Create and return a log file path for monitoring before execution starts."""
        log_file = tempfile.NamedTemporaryFile(
            mode='w+', 
            suffix='.log', 
            prefix='executor_', 
            delete=False,
            encoding='utf-8'
        )
        log_file_path = Path(log_file.name)
        log_file.write("Waiting for execution to start...\n")
        log_file.close()
        return log_file_path

    def _ask_llm_decision(
        self,
        code: str,
        goal: str,
        output: str,
        run_time: float
    ):
        remaining_time = self.cfg.execution_timeout - run_time
        runtime_minutes = run_time / 60
        max_runtime_minutes = self.cfg.execution_timeout / 60
        remaining_minutes = remaining_time / 60

        lines = output.splitlines()
        if len(lines) > 500:
            output = '\n'.join(lines[:250] + ["... (truncated) ..."] + lines[-250:])

        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant monitoring code execution. 
Your task is to analyze the current execution output and decide whether the code should continue running.

Consider these factors:
1. Is the loss exploding (becoming very large or NaN)?
2. Is the loss decreasing normally over time?
3. Are there any error messages indicating failure?
4. Does the output suggest normal training/execution progress?
5. Based on current progress and remaining time, is it possible to complete within the time limit?

Respond in the following format:

<action>
CONTINUE/STOP
</action>

<explanation>
Your rationale for the action. Describe the current progress, your estimated remaining time, and explain why you think the execution should continue or stop. DO NOT GIVE SUGGESTIONS ON BUG FIXES.
</explanation>

."""
            },
            {
                "role": "user", 
                "content": f"""
Code being executed:
```python
{code}
```

Goal: {goal}

Runtime Information:
- Current runtime: {runtime_minutes:.2f} minutes ({run_time:.1f} seconds)
- Maximum runtime: {max_runtime_minutes:.2f} minutes ({self.cfg.execution_timeout} seconds)
- Remaining time: {remaining_minutes:.2f} minutes ({remaining_time:.1f} seconds)

Current output:
```
{output}
```

Based on the current progress, runtime information, and output analysis, should the execution continue? 
Consider whether the task can realistically be completed within the remaining time.

Respond in the following format:

<action>
CONTINUE/STOP
</action>

<explanation>
Your rationale for the action. Describe the current progress, your estimated remaining time, and explain why you think the execution should continue or stop.
</explanation>
"""
            }
        ]

        response = query_llm(self.cfg.llm, messages, required_fields=["action", "explanation"])
        
        should_continue = "continue" in response["action"][0].lower()
        explanation = response["explanation"][0] if response["explanation"] else "No explanation provided"
        
        return should_continue, explanation

    def _write_to_log_file(self, message: str):
        with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(message)

    def _collect_iopub_outputs(self, msg_id: str, code: str, goal: str) -> ExecutionResult:
        self._write_to_log_file("execution started...")
        output, error, success, llm_terminated, timeout = "", "", True, False, False
        start_time = time()
        last_decision_time = start_time
        while True:
            current_time = time()
            if current_time - last_decision_time > self.cfg.execution_inspect_interval:
                last_decision_time = current_time
                should_continue, explanation = self._ask_llm_decision(
                    code, goal, self._process_backspace_chars(output), current_time - start_time
                )
                if not should_continue:
                    llm_terminated = True
                    error = explanation
                    break
            
            if current_time - start_time > self.cfg.execution_timeout:
                timeout = True
                break

            try:
                msg = self.kc.get_iopub_msg(timeout=10)
            except Exception:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "status":
                if content.get("execution_state") == "idle":
                    break
            elif msg_type == "stream":
                output += content.get("text", "")
            elif msg_type == "error":
                success = False
                for line in content.get("traceback", []):
                    error += line + "\n"

            self._write_to_log_file(self._process_backspace_chars(output + "\n" + error))

        if timeout | llm_terminated:
            success = False

        execution_time = time() - start_time
        output, error = self._process_backspace_chars(output), self._process_backspace_chars(error)
        return ExecutionResult(
            success=success,
            llm_terminated=llm_terminated,
            timeout=timeout,
            output=output,
            error=error,
            execution_time=execution_time
        )

    def _process_backspace_chars(self, text: str) -> str:
        """Process backspace characters to show final state of progress bars.
        
        Args:
            text: Raw text with potential backspace characters
            
        Returns:
            Processed text with backspace characters handled
        """
        if not text:
            return text
        
        # Remove ANSI escape sequences (used by tqdm and other progress bars)
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # Split text into lines
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Handle carriage return within line
            if '\r' in line:
                parts = line.split('\r')
                # Keep only the last part (final state)
                line = parts[-1]
            
            # Handle backspace characters
            if '\b' in line:
                result = []
                for char in line:
                    if char == '\b':
                        if result:
                            result.pop()
                    else:
                        result.append(char)
                line = ''.join(result)
            
            processed_lines.append(line)
        
        # Filter out duplicate tqdm progress lines
        # Keep only the last occurrence of each type of progress line
        final_lines = []
        seen_progress_patterns = {}
        
        for line in reversed(processed_lines):
            line_stripped = line.strip()
            
            # Check if this is a tqdm progress line
            if ('|' in line_stripped and '%' in line_stripped and 
                ('it/s' in line_stripped or 's/it' in line_stripped)):
                # Extract the description part (before the percentage)
                desc_match = re.match(r'^([^:]*?):\s*\d+%', line_stripped)
                if desc_match:
                    desc = desc_match.group(1)
                    if desc not in seen_progress_patterns:
                        seen_progress_patterns[desc] = True
                        final_lines.append(line)
                else:
                    # Generic progress pattern
                    if 'generic_progress' not in seen_progress_patterns:
                        seen_progress_patterns['generic_progress'] = True
                        final_lines.append(line)
            elif line_stripped:  # Non-empty non-progress line
                final_lines.append(line)
        
        return '\n'.join(reversed(final_lines))

    def append_cell(self, code: str, goal: str) -> ExecutionResult:
        msg_id = self.kc.execute(code, allow_stdin=False, store_history=True, stop_on_error=True)
        result = self._collect_iopub_outputs(msg_id, code, goal)
        if result.timeout or result.llm_terminated:
            self.km.interrupt_kernel()
        return result

if __name__ == "__main__":
    cfg = Config(execution_timeout=5)
    session = JupyterSession(cfg=cfg, kernel_name="python3")
    result = session.append_cell("""
from time import sleep
from tqdm import tqdm
for i in tqdm(range(10)):
    sleep(2)
    print(i)
""", "")
    print(result)
    session.shutdown()
    del session

