import tempfile
import atexit
import json
import os
import psutil
import subprocess
from time import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from jupyter_client import KernelManager, BlockingKernelClient
from jupyter_client.kernelspec import KernelSpecManager
from comind.community import Draft
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

@dataclass 
class CodeCell:
    code: str
    success: bool

class JupyterNotebook:
    cells: list[CodeCell]

    def __init__(self, cfg: Config, cells: list[CodeCell]):
        self.cfg = cfg
        self.cells = cells

    def append(self, code: str, success: bool):
        self.cells.append(CodeCell(code=code, success=success))

    def get_notebook_code(self) -> str:
        cells_str = ""
        for cell in self.cells:
            if cell.success:
                cells_str += f"<cell>\n{cell.code}\n</cell>\n\n"

        prompt = f"""
You are tasked with converting a Jupyter notebook into clean, production-ready Python code. Your goal is to extract only the essential code that contributes to the final results. 
You should only retain code from cells that were actually executed and required by later cells that produce final results. Remove any unnecessary code and eliminate deprecate code that was replaced by later versions and duplicate implementations where only the final version is used. You extraction should be on cell-level. You should only determine each cell should be retained and rearrange the order of them. Do not change the content of each cell. You should provide a self-contained full python code that can be run as a standalone .py file. Your final code should be contained in a single code tag.

{cells_str}

Now, please provide the assembled python file in the following format:

<summary>
Your summary of the code. Describe the pipeline and model structures.
</summary>

<code>
The assembled python code. Do not include any additional text or explanations. Do not wrap the code in a markdown code block.
</code>
"""
        def check_fn(response: dict):
            return len(response["code"]) == 1
        
        result = query_llm(self.cfg.llm, [{"role": "system", "content": prompt}], required_fields=["code"], check_fn=check_fn)
        return result["code"][0]


class JupyterSession:
    def __init__(self, cfg: Config, env_name: str):
        self.cfg = cfg
        self.cells = JupyterNotebook(cfg=cfg, cells=[])
        self.env_name = env_name
        
        # Select optimal GPU and CPU resources
        self.selected_gpu_ids, self.selected_cpu_cores = self._select_optimal_resources()
        
        kernel_name = self._setup_conda_environment()

        self.km: KernelManager = KernelManager(kernel_name=kernel_name)
        self.km.start_kernel(cwd=self.cfg.agent_workspace_dir)
        self.kc: BlockingKernelClient = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready()
        self.log_file_path = self._get_log_file_path()

        atexit.register(self._safe_shutdown)
    
    def shutdown(self):
        self._safe_shutdown()

    def _get_gpu_memory_usage(self) -> List[Tuple[int, float]]:
        """Get GPU memory usage for all available GPUs.
        
        Returns:
            List of tuples (gpu_id, memory_usage_percentage)
        """
        try:
            # Try using nvidia-ml-py3 first (more reliable)
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                gpu_usage = []
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage_percent = (mem_info.used / mem_info.total) * 100
                    gpu_usage.append((i, usage_percent))
                
                return gpu_usage
            except ImportError:
                # Fallback to nvidia-smi command
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True
                )
                
                gpu_usage = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        gpu_id = int(parts[0])
                        used_mem = float(parts[1])
                        total_mem = float(parts[2])
                        usage_percent = (used_mem / total_mem) * 100
                        gpu_usage.append((gpu_id, usage_percent))
                
                return gpu_usage
                
        except Exception as e:
            print(f"Warning: Could not get GPU information: {e}")
            return []

    def _get_cpu_core_utilization(self) -> List[Tuple[int, float]]:
        """Get CPU core utilization for all available cores.
        
        Returns:
            List of tuples (core_id, utilization_percentage)
        """
        try:
            # Get per-core CPU utilization over a short interval
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            return [(i, usage) for i, usage in enumerate(cpu_percent)]
        except Exception as e:
            print(f"Warning: Could not get CPU information: {e}")
            # Fallback: return all cores with 0% usage
            return [(i, 0.0) for i in range(psutil.cpu_count())]

    def _select_optimal_resources(self) -> Tuple[List[int], List[int]]:
        """Select optimal GPU and CPU resources based on current usage.
        
        Returns:
            Tuple of (selected_gpu_ids, selected_cpu_cores)
        """
        # Get current CUDA_VISIBLE_DEVICES if set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        available_gpus = []
        
        if cuda_visible:
            try:
                available_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            except ValueError:
                available_gpus = []
        
        # Get GPU usage and select best ones
        selected_gpu_ids = []
        gpu_usage = self._get_gpu_memory_usage()
        
        if gpu_usage:  # If we can detect GPUs
            if available_gpus:
                # Filter to only available GPUs from CUDA_VISIBLE_DEVICES
                available_gpu_usage = [(gpu_id, usage) for gpu_id, usage in gpu_usage if gpu_id in available_gpus]
            else:
                # If CUDA_VISIBLE_DEVICES is empty or not set, use all detected GPUs
                available_gpu_usage = gpu_usage
            
            if available_gpu_usage:
                # Sort by memory usage (ascending) and select the least used ones
                available_gpu_usage.sort(key=lambda x: x[1])
                selected_gpu_ids = [gpu_id for gpu_id, _ in available_gpu_usage[:self.cfg.execution_max_gpu_count]]
        
        # Get CPU usage and select best cores
        cpu_usage = self._get_cpu_core_utilization()
        # Sort by utilization (ascending) and select the least used ones
        cpu_usage.sort(key=lambda x: x[1])
        selected_cpu_cores = [core_id for core_id, _ in cpu_usage[:self.cfg.execution_max_cpu_cores]]
        
        print(f"Selected GPU IDs: {selected_gpu_ids}")
        print(f"Selected CPU cores: {selected_cpu_cores}")
        
        return selected_gpu_ids, selected_cpu_cores

    def _setup_conda_environment(self) -> str:
        # Find the conda environment path for this workspace
        conda_env_path = self.cfg.conda_envs_dir
        
        conda_python = conda_env_path.absolute() / self.env_name / "bin" / "python"
        assert conda_python.exists(), f"Conda environment python not found at {conda_python}"
        
        kname = self.env_name
        ksm = KernelSpecManager()
        
        # Prepare environment variables for resource limiting
        kernel_env = {}
        
        # Set CUDA_VISIBLE_DEVICES to selected GPUs
        if self.selected_gpu_ids:
            kernel_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.selected_gpu_ids))
        else:
            # If no GPUs selected, hide all GPUs
            kernel_env["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set CPU affinity using taskset if available (Linux)
        kernel_argv = [str(conda_python), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        
        # On Linux, use taskset to limit CPU cores
        if self.selected_cpu_cores and os.name == 'posix':
            try:
                # Check if taskset is available
                subprocess.run(['taskset', '--version'], capture_output=True, check=True)
                cpu_mask = ",".join(map(str, self.selected_cpu_cores))
                kernel_argv = ["taskset", "-c", cpu_mask] + kernel_argv
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: taskset not available, CPU core limiting will not be applied")
        
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            kernel_spec = {
                "argv": kernel_argv,
                "display_name": kname,
                "language": "python",
            }
            
            # Add environment variables if any
            if kernel_env:
                kernel_spec["env"] = kernel_env
                
            (td / "kernel.json").write_text(json.dumps(kernel_spec, indent=2), encoding="utf-8")
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
        self.cells.append(code=code, success=result.success)
        return result
    
    def get_notebook_code(self) -> str:
        return self.cells.get_notebook_code()
        


