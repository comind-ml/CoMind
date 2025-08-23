import asyncio
import os
import re
import subprocess
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import psutil

from comind.config import Config, LLMConfig
from comind.utils.llm import query_llm


@dataclass
class ExecutionResult:
    """Class to collect execution results."""
    terminated_unexpectedly: bool
    timeout: bool
    execution_time: float
    final_output: str
    success: bool
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    llm_termination_reason: Optional[str] = None
    log_file_path: Optional[Path] = None


def get_gpu_memory_usage() -> List[tuple[int, float]]:
    """Get GPU memory usage for GPUs listed in CUDA_VISIBLE_DEVICES.
    
    Returns:
        List of tuples (gpu_id, memory_used_mb) sorted by memory usage (ascending)
    """
    # First, get the list of visible GPUs from environment variable
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    visible_gpu_ids = []
    
    if cuda_visible:
        try:
            visible_gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            print(f"Using GPUs from CUDA_VISIBLE_DEVICES: {visible_gpu_ids}")
        except ValueError:
            print(f"Warning: Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible}")
            visible_gpu_ids = []
    
    # If no CUDA_VISIBLE_DEVICES set, detect and use all available GPUs
    if not visible_gpu_ids:
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
            gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
            visible_gpu_ids = list(range(gpu_count))
            print(f"No CUDA_VISIBLE_DEVICES set, using all available GPUs: {visible_gpu_ids}")
        except Exception:
            print("Warning: Could not detect GPUs, assuming single GPU (ID: 0)")
            visible_gpu_ids = [0]
    
    try:
        # Try nvidia-smi to get memory usage for visible GPUs only
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(', ')
                if len(parts) >= 2:
                    gpu_id = int(parts[0])
                    memory_used = float(parts[1])
                    # Only include GPUs that are in CUDA_VISIBLE_DEVICES
                    if gpu_id in visible_gpu_ids:
                        gpu_memory.append((gpu_id, memory_used))
        
        # Sort by memory usage (ascending - least used first)
        gpu_memory.sort(key=lambda x: x[1])
        return gpu_memory
        
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get GPU memory usage via nvidia-smi: {e}")
        
        # Fallback: return visible GPUs with 0 memory usage (unknown)
        return [(gpu_id, 0.0) for gpu_id in visible_gpu_ids]


def select_least_utilized_gpus(max_gpu_count: int) -> str:
    """Select the GPUs with the lowest memory usage.
    
    Args:
        max_gpu_count: Maximum number of GPUs to select
        
    Returns:
        Comma-separated string of GPU IDs (e.g., "0,1,2")
    """
    try:
        gpu_memory = get_gpu_memory_usage()
        
        if not gpu_memory:
            print("Warning: No GPUs detected, using GPU 0")
            return "0"
        
        # Select the requested number of least utilized GPUs
        num_gpus_to_select = min(max_gpu_count, len(gpu_memory))
        selected_gpus = [str(gpu_id) for gpu_id, _ in gpu_memory[:num_gpus_to_select]]
        
        # Print selection info
        selected_info = [(gpu_id, memory) for gpu_id, memory in gpu_memory[:num_gpus_to_select]]
        print(f"Selected GPUs with lowest memory usage: {selected_info}")
        
        return ','.join(selected_gpus)
        
    except Exception as e:
        print(f"Warning: GPU selection failed: {e}")
        # Fallback to first available GPUs
        fallback_gpus = [str(i) for i in range(min(max_gpu_count, 1))]
        return ','.join(fallback_gpus)


class Executor:
    """Executor class for running code with monitoring and LLM-based decision making."""
    
    def __init__(self, cfg: Config):
        """Initialize the Executor with configuration.
        
        Args:
            cfg: Configuration object containing workspace and LLM settings
        """
        self.cfg = cfg
        self.workspace = Path(cfg.agent_workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def get_log_file_path(self) -> Path:
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
        
    def execute(
        self,
        code: str,
        goal: str,
        log_file_path: Optional[Path] = None,
    ) -> ExecutionResult:
        """Execute code with monitoring and LLM decision making.
        
        Args:
            code: Python code to execute
            goal: Goal description for the execution
            inspect_interval: Interval in seconds to check terminal output
            timeout: Maximum execution time in seconds
            max_cpu_cores: Maximum CPU cores to use
            max_gpu_count: Maximum number of GPUs to use (dynamically selected based on memory usage)
            
        Returns:
            ExecutionResult containing execution details
        """
        if self.cfg.execution_max_gpu_count <= 0:
            raise ValueError("max_gpu_count must be greater than 0")
            
        # Write code to agent.py in workspace
        agent_file = self.workspace / "agent.py"
        agent_file.write_text(code)
        
        start_time = time.time()
        output_buffer = []
        process = None
        terminated_unexpectedly = False
        llm_termination_reason = None
        timeout_occurred = False
        
        # Use provided log file path or create a new one
        if log_file_path is None:
            log_file = tempfile.NamedTemporaryFile(
                mode='w+', 
                suffix='.log', 
                prefix='executor_', 
                delete=False,
                encoding='utf-8'
            )
            log_file_path = Path(log_file.name)
            log_file.write("Execution starting...\n")
            log_file.flush()
        else:
            # Open the existing log file
            log_file = open(log_file_path, 'w', encoding='utf-8')
            log_file.write("Execution starting...\n")
            log_file.flush()
        
        try:
            # Dynamically select GPUs based on memory usage
            selected_gpus = select_least_utilized_gpus(self.cfg.execution_max_gpu_count)
            
            # Set up environment with GPU and CPU constraints
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = selected_gpus
            
            # Start the process
            process = subprocess.Popen(
                ['python', 'agent.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                cwd=str(self.workspace)
            )
            
            # Apply CPU affinity if specified
            if self.cfg.execution_max_cpu_cores > 0:
                try:
                    p = psutil.Process(process.pid)
                    selected_cpus = self._select_least_utilized_cpus(self.cfg.execution_max_cpu_cores)
                    p.cpu_affinity(selected_cpus)
                except Exception as e:
                    print(f"Warning: Could not set CPU affinity: {e}")
            
            last_check_time = start_time
            last_log_write_time = start_time
            
            while process.poll() is None:
                current_time = time.time()
                
                # Check for timeout
                if current_time - start_time > self.cfg.execution_timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    terminated_unexpectedly = True
                    timeout_occurred = True
                    break
                
                # Read available output (character by character for real-time progress bars)
                try:
                    # Use select to check if data is available
                    import select
                    if select.select([process.stdout], [], [], 0)[0]:
                        # Read available data
                        data = process.stdout.read(1024)  # Read in chunks
                        if data:
                            output_buffer.append(data)
                except Exception:
                    # Fallback to line-based reading
                    try:
                        if process.stdout.readable():
                            line = process.stdout.readline()
                            if line:
                                output_buffer.append(line.rstrip())
                    except Exception:
                        pass
                
                # Write processed output to log file periodically (every 2 seconds)
                if current_time - last_log_write_time >= 2.0:
                    last_log_write_time = current_time
                    try:
                        # Process output to handle backspace characters
                        raw_output = ""
                        for item in output_buffer:
                            if isinstance(item, bytes):
                                raw_output += item.decode('utf-8', errors='ignore')
                            else:
                                raw_output += str(item)
                        
                        # Process backspace characters to get final terminal display
                        processed_output = self._process_backspace_chars(raw_output)
                        
                        # Write processed output to log file (overwrite each time)
                        log_file.seek(0)
                        log_file.truncate()
                        log_file.write(processed_output)
                        log_file.flush()
                    except Exception as e:
                        print(f"Error writing to log file: {e}")
                        pass
                
                # Check if it's time for LLM inspection
                if current_time - last_check_time >= self.cfg.execution_inspect_interval:
                    last_check_time = current_time
                    
                    # Process output to handle backspace characters
                    # Convert buffer to string, handling both bytes and strings
                    raw_output = ""
                    for item in output_buffer:
                        if isinstance(item, bytes):
                            raw_output += item.decode('utf-8', errors='ignore')
                        else:
                            raw_output += str(item)
                    
                    processed_output = self._process_backspace_chars(raw_output)
                    
                    # Ask LLM for decision
                    should_continue, llm_explanation = self._ask_llm_decision(
                        code, 
                        self.cfg.competition_task_desc or "", 
                        goal, 
                        processed_output,
                        current_time - start_time,
                        self.cfg.execution_timeout
                    )
                    
                    if not should_continue:
                        llm_termination_reason = llm_explanation
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        terminated_unexpectedly = True
                        break
                
                # Small sleep to prevent busy waiting
                time.sleep(1)
            
            # Read any remaining output
            if process.stdout:
                try:
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        if isinstance(remaining_output, bytes):
                            output_buffer.append(remaining_output)
                        else:
                            output_buffer.append(remaining_output)
                except Exception:
                    pass
            
            execution_time = time.time() - start_time
            
            # Convert all output to string and process
            raw_output = ""
            for item in output_buffer:
                if isinstance(item, bytes):
                    raw_output += item.decode('utf-8', errors='ignore')
                else:
                    raw_output += str(item)
            
            final_output = self._process_backspace_chars(raw_output)
            
            # Write final processed output to log file
            try:
                log_file.seek(0)
                log_file.truncate()
                log_file.write(final_output)
                log_file.flush()
            except Exception as e:
                print(f"Error writing final output to log file: {e}")
            
            exit_code = process.returncode if process else None
            success = exit_code == 0 if exit_code is not None else False
            
            return ExecutionResult(
                terminated_unexpectedly=terminated_unexpectedly,
                timeout=timeout_occurred,
                execution_time=execution_time,
                final_output=final_output,
                success=success,
                exit_code=exit_code,
                llm_termination_reason=llm_termination_reason,
                log_file_path=log_file_path
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                terminated_unexpectedly=True,
                timeout=False,
                execution_time=execution_time,
                final_output='\n'.join(output_buffer),
                success=False,
                error_message=str(e),
                llm_termination_reason=llm_termination_reason,
                log_file_path=log_file_path
            )
        finally:
            # Cleanup: ensure process is terminated
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception:
                    pass
            
            # Close log file
            try:
                log_file.close()
            except Exception:
                pass
    
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
    
    def _select_least_utilized_cpus(self, max_cpu_cores: int) -> list[int]:
        """Select the CPUs with the lowest utilization.
        
        Args:
            max_cpu_cores: Maximum number of CPU cores to select
            
        Returns:
            List of CPU indices with lowest utilization
        """
        try:
            # Get per-CPU utilization over a short interval
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Create list of (cpu_index, utilization) tuples
            cpu_utilization = [(i, usage) for i, usage in enumerate(cpu_percent)]
            
            # Sort by utilization (ascending - lowest first)
            cpu_utilization.sort(key=lambda x: x[1])
            
            # Select the requested number of least utilized CPUs
            num_cpus_to_select = min(max_cpu_cores, len(cpu_utilization))
            selected_cpus = [cpu_idx for cpu_idx, _ in cpu_utilization[:num_cpus_to_select]]
            
            print(f"Selected CPUs with lowest utilization: {selected_cpus}")
            return selected_cpus
            
        except Exception as e:
            print(f"Warning: Could not determine CPU utilization, falling back to first {max_cpu_cores} CPUs: {e}")
            # Fallback to first available CPUs
            return list(range(min(max_cpu_cores, psutil.cpu_count())))
    
    def _ask_llm_decision(
        self, 
        code: str, 
        task_desc: str, 
        goal: str, 
        current_output: str,
        current_runtime: float,
        max_runtime: int
    ) -> tuple[bool, str]:
        """Ask LLM to decide whether to continue execution.
        
        Args:
            code: The code being executed
            task_desc: Task description from config
            goal: Execution goal
            current_output: Current terminal output
            current_runtime: Current runtime in seconds
            max_runtime: Maximum allowed runtime in seconds
            
        Returns:
            Tuple of (should_continue, explanation) where:
            - should_continue: True if execution should continue, False otherwise
            - explanation: LLM's explanation for the decision
        """
        try:
            remaining_time = max_runtime - current_runtime
            runtime_minutes = current_runtime / 60
            max_runtime_minutes = max_runtime / 60
            remaining_minutes = remaining_time / 60

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
Your rationale for the action. Describe the current progress, your estimated remaining time, and explain why you think the execution should continue or stop.
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

Task description: {task_desc}

Goal: {goal}

Runtime Information:
- Current runtime: {runtime_minutes:.2f} minutes ({current_runtime:.1f} seconds)
- Maximum runtime: {max_runtime_minutes:.2f} minutes ({max_runtime} seconds)
- Remaining time: {remaining_minutes:.2f} minutes ({remaining_time:.1f} seconds)

Current output (last 100000 characters):
```
{current_output[-100000:]}
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
            
        except Exception as e:
            print(f"Warning: LLM decision failed: {e}")
            # Default to continue if LLM fails
            return True, f"LLM decision failed: {str(e)}"

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    
    # Create proper config with all required attributes
    llm_config = LLMConfig(
        model="gpt-4",
        params={"temperature": 0.1},
        max_retries=3
    )
    
    config = Config(
        competition_task_desc="Test execution with progress bar",
        llm=llm_config,
        agent_workspace_dir=Path(tempfile.mkdtemp()),
        execution_timeout=10
    )
    
    executor = Executor(config)
    
    print("🧪 Testing Executor with progress bar and LLM monitoring...")
    print("=" * 60)
    
    result = executor.execute(
        code="""
import time
import sys
from tqdm import tqdm

print("Starting progress bar test...")

# Test with tqdm progress bar - longer execution to trigger LLM
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.8)  # Longer sleep to ensure LLM inspection

print("\\nProgress bar completed!")

# Additional steps to extend execution time
print("Running additional processing...")
for i in range(3):
    print(f"Additional step {i+1}/3")
    time.sleep(1)

print("Final result: Test successful")
""",
        goal="Test progress bar processing and LLM monitoring",
    )
    
    print("\\n" + "=" * 60)
    print("RESULTS:")
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Terminated unexpectedly: {result.terminated_unexpectedly}")
    print(f"Timeout: {result.timeout}")
    print(f"LLM termination reason: {result.llm_termination_reason}")
    print(f"Exit code: {result.exit_code}")
    print(f"Error message: {result.error_message}")
    
    print("\\nFINAL OUTPUT:")
    print("-" * 40)
    print(result.final_output)
    print("-" * 40)