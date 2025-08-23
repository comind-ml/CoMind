import time
import pickle
import html
import re
import threading
from pathlib import Path
from comind.config import Config
from comind.community import Draft, Pipeline
from comind.utils import Conversation, Executor, ExecutionResult, generate, WorstMetricValue, query_llm, MetricValue
from comind.agent import MetricUpdater

class CodeAgent:
    def __init__(self, cfg: Config, draft: Draft, is_lower_better: bool, metric_updater: MetricUpdater):
        self.cfg = cfg
        self.draft = draft
        self.llm = Conversation(cfg.llm)
        self.executor = Executor(cfg)
        self.start_time = time.time()
        self.iteration = 0
        self.best_metric = WorstMetricValue()
        self.is_lower_better = is_lower_better
        self.metric_updater = metric_updater
        
        # Initialize coder state for monitoring
        self.messages = []
        self.current_code = draft.code
        self.output_lines = ["Unavailabe. The coder is initializing..."]
        self.state_file = self.cfg.agent_workspace_dir.parent / "coder_state.pkl"
        
        # Execution monitoring
        self._execution_thread = None
        self._execution_active = False
        self._execution_result = None
        self._real_time_output = []
        self._execution_lock = threading.Lock()
        self._output_monitor_thread = None
        self._current_log_file = None
        
    def _save_state(self):
        """Save current coder state for monitoring."""
        try:
            coder_state = {
                'name': self.draft.title,
                'messages': self.messages.copy(),
                'code': self.current_code,
                'output_lines': self.output_lines.copy(),
                'iteration': self.iteration,
                'best_metric': str(self.best_metric),
                'start_time': self.start_time,
                'draft_id': self.draft.id,
                'is_running': True  # Indicate this coder is actively running
            }
            
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(coder_state, f)
                
        except Exception as e:
            # Don't let state saving errors crash the coder
            print(f"Warning: Failed to save coder state: {e}")
    
    def _add_message(self, role: str, content: str):
        """Add a message and update state."""
        # replace <code>...</code> with ```python\ncode\n```
        content = re.sub(r'<code>([\s\S]*?)</code>', r'```python\n\1\n```', content)
        
        # Escape HTML tags except within code blocks
        parts = content.split("```")
        for i in range(0, len(parts), 2):
            # Only escape non-code parts (even indices)
            parts[i] = html.escape(parts[i])
        content = "```".join(parts)
        
        # Actually add the message to the list
        self.messages.append((role, content))
        self._save_state()
    
    def _update_code(self, code: str):
        """Update current code and save state."""
        self.current_code = code
        self._save_state()
    
    def _update_output(self, output_lines: list):
        """Update output lines and save state."""
        self.output_lines = output_lines
        self._save_state()
    
    def _start_output_monitor(self, log_file_path=None):
        """Start monitoring real-time output from executor."""
        if self._output_monitor_thread and self._output_monitor_thread.is_alive():
            return
            
        def monitor_output():
            last_modified = 0
            last_content_hash = ""
            
            while self._execution_active:
                try:
                    # Use the log file provided by executor
                    if self._current_log_file and self._current_log_file.exists():
                        # Check file modification time instead of size
                        current_modified = self._current_log_file.stat().st_mtime
                        
                        # Always read and check content since executor overwrites the file
                        try:
                            with open(self._current_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                full_content = f.read()
                            
                            # Use content hash to detect actual changes
                            import hashlib
                            content_hash = hashlib.md5(full_content.encode()).hexdigest()
                            
                            if content_hash != last_content_hash or current_modified > last_modified:
                                with self._execution_lock:
                                    # Update output lines with full content (no truncation)
                                    self.output_lines = full_content.split('\n') if full_content else ["No output yet..."]
                                    self._save_state()
                                    print(f"🔄 Output updated: {len(self.output_lines)} lines")
                                
                                last_content_hash = content_hash
                                last_modified = current_modified
                        except Exception as read_error:
                            print(f"Error reading log file: {read_error}")
                    else:
                        # Log file doesn't exist yet, update with placeholder
                        with self._execution_lock:
                            self.output_lines = ["Execution starting, waiting for output..."]
                            self._save_state()
                    
                    time.sleep(20)  # Check every 20 seconds
                except Exception as e:
                    print(f"Error in output monitor: {e}")
                    break
                    
        self._output_monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        self._output_monitor_thread.start()
    
    def _stop_output_monitor(self):
        """Stop the output monitor thread."""
        self._execution_active = False
        if self._output_monitor_thread and self._output_monitor_thread.is_alive():
            self._output_monitor_thread.join(timeout=5)
    
    def _execute_async(self, code: str, goal: str):
        """Execute code asynchronously with real-time monitoring."""
        def execute_in_background():
            try:
                with self._execution_lock:
                    self._execution_active = True
                    self._real_time_output = ["Starting execution..."]
                    self._execution_result = None
                
                # Create a custom executor that allows us to get the log file early
                # We'll use a modified version of the execute method
                result = self._execute_with_monitoring(code, goal)
                
                with self._execution_lock:
                    self._execution_result = result
                    # Update final output
                    if result:
                        self.output_lines = result.final_output.split('\n')
                    else:
                        self.output_lines = ["Execution failed or timed out"]
                    self._save_state()
                    
            except Exception as e:
                with self._execution_lock:
                    self._real_time_output.append(f"Execution error: {str(e)}")
                    self.output_lines = self._real_time_output.copy()
                    self._save_state()
            finally:
                self._stop_output_monitor()
                
        self._execution_thread = threading.Thread(target=execute_in_background, daemon=True)
        self._execution_thread.start()
    
    def _execute_with_monitoring(self, code: str, goal: str):
        """Execute code with real-time monitoring using executor's log file."""
        try:
            # Get log file path before execution starts
            log_file_path = self.executor.get_log_file_path()
            
            # Set the log file path and start monitoring immediately
            with self._execution_lock:
                self._current_log_file = log_file_path
            
            # Start monitoring the log file
            self._start_output_monitor()
            
            print(f"🔄 Starting execution with log file: {log_file_path}")
            
            # Execute using the executor with our log file
            result = self.executor.execute(code, goal, log_file_path)
            
            return result
        except Exception as e:
            print(f"Execution error: {e}")
            return None
    
    def _wait_for_execution(self, timeout=None):
        """Wait for execution to complete and return the result."""
        if self._execution_thread:
            self._execution_thread.join(timeout=timeout)
            
        with self._execution_lock:
            return self._execution_result
    
    def _get_packages(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        return pkgs

    def _post_initial_message(self):
        data_preview = generate((self.cfg.agent_workspace_dir.parent / "input"))
        prompt = f"""
You're an expert Kaggle competitor tasked with implementing a pipeline into Python code. You can modify the details (training parameters, feature engineering, model selection, etc. ), but do not change overall architecture of this pipeline. The goal is to **obtain best score** on this competition.

<task_desc>\n{self.cfg.competition_task_desc}\n</task_desc>

<pipeline>\n{self.draft.description}\n</pipeline>

This is the code abstract of the pipeline:
<code>\n{self.draft.code}\n</code>

All the input files are visible in ../input folder, this folder typically contains the competition data and external resouces, including public datasets, models and outputs of other kernels. DO NOT USE /kaggle/input paths in your code. USE ../input instead. Here is an abstract of the input files (../input):

<data_preview>\n{data_preview}\n</data_preview>
"""
        if self.draft.codebase_content is not None:
            prompt += f"""
You will develop the pipeline based on this codebase. Any output files of the codebase, such as csvs, checkpoints, etc., are visible in ./, which is your current working directory. 
<codebase_content>\n{self.draft.codebase_content}\n</codebase_content>
"""

        prompt += f"""
Your code must produce a submission at ./submission.csv, this is EXTREMELY IMPORTANT. Before generating the submission, you should print the value of the evaluation metric computed on a hold-out validation set. You can use custom evaluation functions during training, but the final metric **MUST FOLLOW THE EVALUATION SECTION IN THE TASK DESCRIPTION** on a validation set. If other kernels with submission.csv are provided in the input folder, you can ensemble them before generating your own submission. This is important because we will pick your best code based on this metric. You are allowed to load the checkpoints of other models. Do not contain any absolute paths in your code. Time limit per run is 2 hours. Your code will be killed if timeout. 

Your code will be executed on a single A6000 GPU. Use large batchsizes to maximize the gpu utilization. If the code segment is provided in this prompt, you should follow the input/output structure. You are allowed to install any packages you need or inspect the workspace (e.g., print file contents, check folder structure). DO NOT USE ABSOLUTE PATHS IN YOUR CODE.

The workspace will be maintained across iterations. That is, if your first iteration code produces a checkpoint, you can load it in the second iteration. You can ensemble submissions generated by yourself and other kernels. You should generate model checkpoints for future loading. If you load the external submissions successfully but failed to merge them with your own predictions, you should print the headers of the external submission and your own predictions and check if the ids are aligned. All the external submissions are valid. Your predictions should be in the same format as them.

We have installed the following packages: {", ".join(self._get_packages())}. You are allowed to install any packages by running `pip install <package_name>` in your script.

Now, please generate the full code based on the above instructions. You must respond in the following format:

<goal>
The explanation of your code. You should describe the desired exeuction time and output of your code. Explain any modifications you made and how to interpret the output. 
</goal>

<code>
The full python code. This should be a self-contained script other than a code segment. propose full code even if it is a minor change. Do not wrap the code in a markdown block. Your code will be stored in ./code.py and executed by `python code.py`.
</code>

Remember, all input files are stored in ../input folder. DO NOT USE /kaggle/input paths or other absolute paths in your code. Now, propose your first implementation.
"""
        self.llm.add_message(role="system", content=prompt)
        self._add_message("agent", prompt)

    def _generate_report(self) -> Pipeline:
        if isinstance(self.best_metric, WorstMetricValue):
            prompt = f"""
Your code failed to produce a submission. Please summarize your code and explain why it failed. You should include the difficulties you encountered and possible bugs.
"""
        else:
            prompt = f"""
The best metric value across all iterations is {self.best_metric}. Please summarize the code that achieved this metric value. Explain your code with the best score and the modifications you made to the codebase (if the codebase is provided).
"""
        prompt += f"""
You should respond in the following format:

<description>
An extremely detailed description of the pipeline that generated the best results. All hyperparameters, training settings, model architectures, feature engineering, validation metric, and any other relevant information should be included. Describe potential improvements and future work. Include other parts mentioned above. Describe the checkpoint files you generated and how to load them.
</description>

<code>
The abstract of your code. Make sure to include any important parts.
</code>

<suggestions>
An extremely detailed description of the weaknesses of the pipeline you found during the implementation. Determine the bottleneck of the pipeline and suggest possible improvements. You should metion the feasibility, novelty, effectiveness, efficiency of each components of the pipeline, and give a confidence score for your assertions. This should be a markdown list block.
</suggestions>
"""
        self.llm.add_message(role="user", content=prompt)
        self._add_message("agent", prompt)
        report = self.llm.query(required_fields=["description", "code", "suggestions"], check_fn = lambda x: len(x["suggestions"]) == 1)
        self._add_message("llm", report['_raw_content'])
        submission = self.cfg.agent_workspace_dir / "submission.csv"
        if isinstance(self.best_metric, WorstMetricValue):
            submission = None

        # Mark coder as completed and save final state
        try:
            coder_state = {
                'name': self.draft.title,
                'messages': self.messages.copy(),
                'code': self.current_code,
                'output_lines': self.output_lines.copy(),
                'iteration': self.iteration,
                'best_metric': str(self.best_metric),
                'start_time': self.start_time,
                'draft_id': self.draft.id,
                'is_running': False,  # Mark as completed
                'completed': True
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(coder_state, f)
            print(f"🏁 Coder {self.draft.id} completed with final best metric: {self.best_metric}")
        except Exception as e:
            print(f"Warning: Failed to save final coder state: {e}")

        return Pipeline(
            id=self.draft.id,
            title=self.draft.title,
            description=report["description"][0],
            code=report["code"][0],
            referenced_private_data=False,
            metric=self.best_metric,
            submission=submission,
            output_dir=self.cfg.agent_workspace_dir,
            suggestions=report["suggestions"][0],
            datasets=self.draft.datasets,
        )

    def _summarize_result(self, code: str, goal: str, result: ExecutionResult) -> str:
        if len(result.final_output) > 100000:
            result.final_output = result.final_output[:50000] + "\n... (truncated) ... \n" + result.final_output[-50000:]

        prompt = f"""
You are a Kaggle grandmaster attending a competition. You have written code to solve this task and now need to evaluate the output of the code execution. You should determine if there were any bugs as well as report the empirical findings. Include essential information about the result, including warnings, errors, and the final metric. Determine whether the evaluation result is valid. 

<code>
{code} 
</code>

<goal>
{goal}
</goal>

<execution_output>
{result.final_output}
</execution_output>
"""
        if result.terminated_unexpectedly or result.exit_code != 0:
            if result.llm_termination_reason is not None:
                prompt += f"""
The code execution was manually terminated with the following reason:

<reason>
{result.llm_termination_reason}
</reason>
"""
            else:
                if result.timeout:
                    prompt += "\nThe code execution was terminated due to timeout. \n"
                else:
                    prompt += "\nThe code execution was terminated unexpectedly. \n"
            prompt += "Please carefully analyze the execution output and determine the reason for the termination. You should give a suggestion to fix the bug."
        else:
            prompt += "\nThe code execution was successful. Please verify the output of the code execution and report if the evaluation result is valid. Mention if the evaluation metric is calculated properly and match the metric in the task description.\n"

        prompt += f"""
You should respond in the following format:

<abstract>
Select representative segments of the output log and mark the remainder as ellipses. This should contain any critical information, including errors, loss values, final metric values, debug messages, etc.
</abstract>

<summary>
A short summary (4-5 sentences) describing the empirical findings. Examine whether its goals are achieved. Summarize the output and mention if the submission.csv was properly produced. Give suggestions to fix the bug if the code execution was terminated unexpectedly.
</summary>

<metric>
You should report the value of the **final metric** (not the training loss value) if the code execution was successful, producing submission.csv and the metric is calculated properly on the full test set (not dummy or partial). Otherwise, if the code execution was terminated or output messages indicate any training/execution failure, leave it as None. This section should be a real number or None. Report decimal number if the metric is a float.
</metric>
"""

        def check_fn(response):
            metric = response["metric"][0]
            if "none" in metric.lower():
                return True 
            try: 
                metric = float(metric)
                return metric is not None
            except:
                return False

        response = query_llm(self.cfg.llm, messages=[{
            "role": "system",
            "content": prompt
        }], required_fields=["abstract", "summary", "metric"], check_fn=check_fn)

        if "none" in response["metric"][0].lower():
            metric = WorstMetricValue()
        else:
            metric = MetricValue(float(response["metric"][0]), maximize=not self.is_lower_better)

        submission = self.cfg.agent_workspace_dir / "submission.csv"
        if not submission.exists():
            metric = WorstMetricValue()
        
        if not isinstance(metric, WorstMetricValue):
            if metric > self.best_metric:
                old_best = self.best_metric
                self.best_metric = metric
                print(f"🎯 Coder {self.draft.id} best metric updated: {old_best} -> {self.best_metric}")
                # Save state immediately when best metric is updated
                self._save_state()
            self.metric_updater.post(metric, code, submission)
        
        return f"""
<execution_time>
{result.execution_time} seconds
</execution_time>

<timeout>
{result.timeout}
</timeout>

<summary>
{response["summary"][0]}
</summary>

<metric>
{metric}
</metric>

<abstract>
{response["abstract"][0]}
</abstract>
"""
    
    def run(self) -> Pipeline:
        self._post_initial_message()
        for iteration in range(self.cfg.agent_num_iterations_code_agents):
            self.iteration = iteration
            
            time_elapsed = time.time() - self.start_time
            if iteration + 1 == self.cfg.agent_num_iterations_code_agents \
                or time_elapsed > self.cfg.agent_time_limit_code_agents:
                self.llm.pop_message()
                break 
            
            response = self.llm.query(required_fields=["code", "goal"])
            self._add_message("llm", response['_raw_content'])
            self._update_code(response["code"][0])
            
            # Start async execution with real-time monitoring
            self._execute_async(response["code"][0], response["goal"][0])
            
            # Wait for execution to complete (with timeout)
            result = self._wait_for_execution(timeout=self.cfg.execution_timeout)
            
            if result is None:
                # Execution timed out or failed
                result = ExecutionResult(
                    terminated_unexpectedly=True,
                    timeout=True,
                    execution_time=self.cfg.execution_timeout,
                    final_output="Execution timed out or failed",
                    success=False
                )
            
            # Final output update
            output_lines = result.final_output.split('\n')
            self._update_output(output_lines)

            result_summary = self._summarize_result(response["code"][0], response["goal"][0], result)

            prompt = f"""
Remaining steps: {self.cfg.agent_num_iterations_code_agents - iteration - 1}
I ran your code and summarized the execution result:
{result_summary}

Now, please choose your next action and propose code using the same response format as before. Remember, output a self-contained code, no part of it should be omitted. Keep the final validation metric same as the metric mentioned in task description. If your next code will generate checkpoints, **don't give them the same name as previous ones**. If previous code crashed after generating checkpoints and you are fixing the bug, you MUST load the previous checkpoints instead of training from scratch.

A) Fix runtime errors (if any)
B) Do hyperparameter tuning
C) Include ideas that were not implemented yet
D) Add possible improvements
E) Run on a larger scale (moderately increase training epochs, etc.). You should refer to the previous execution time we reported. Remember your code will be killed if timeout.
"""
            self.llm.add_message(role="user", content=prompt)
            self._add_message("agent", prompt)

        return self._generate_report()
        
