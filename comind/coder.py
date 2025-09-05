import time
import pickle
import html
import re
import threading
import shutil
from copy import deepcopy
from comind.config import Config
from comind.community import Draft, Pipeline
from comind.utils import Conversation, generate, WorstMetricValue, MetricValue, get_logger

from comind.utils.jupyter_session import JupyterSession, ExecutionResult
from comind.agent import MetricUpdater
from comind.evaluate import Evaluator

class CodeAgent:
    # ============================================================================
    # INITIALIZATION AND CONFIGURATION
    # ============================================================================
    
    def __init__(
        self, 
        cfg: Config, 
        draft: Draft, 
        is_lower_better: bool, 
        metric_updater: MetricUpdater,
        evaluator: Evaluator | None,
        env_name: str
    ):
        # Core configuration
        self.cfg = cfg
        self.draft = draft
        self.is_lower_better = is_lower_better
        self.evaluator = evaluator
        self.submission_name = cfg.agent_submission_file_name
        self.metric_updater = metric_updater
        self.jupyter_session = JupyterSession(cfg, env_name)
        
        # Timing and iteration tracking
        self.start_time = time.time()
        self.iteration = 0

        # Logging setup
        self.logger = get_logger(f"coder-{draft.id}", self.cfg.agent_workspace_dir / "coder.log")
        self.llm_logger = get_logger(f"llm-{draft.id}", self.cfg.agent_workspace_dir / "llm.log", file_only=True)
        self.llm = Conversation(cfg.llm, logger=self.llm_logger)

        self.logger.info(f"Coder {draft.id} using conda environment {env_name}")

        # State tracking
        self.messages = []
        self.current_code = draft.code
        self.output_lines = ["Unavailable. The coder is initializing..."]
        self.best_code = None
        self.best_metric = WorstMetricValue()
        
        # File paths setup
        (self.cfg.agent_workspace_dir / "best_submission").mkdir(parents=True, exist_ok=True)
        self.best_submission_path = self.cfg.agent_workspace_dir / "best_submission" / self.submission_name
        self.state_file = self.cfg.agent_workspace_dir.parent / "coder_state.pkl"
        
        # Threading support for real-time monitoring
        self._execution_thread = None
        self._execution_active = False
        self._execution_result = None
        self._execution_lock = threading.Lock()
        self._output_monitor_thread = None
        self._current_log_file = None

    def _get_packages(self):
        """Get list of pre-installed packages."""
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

    # ============================================================================
    # STATE MANAGEMENT
    # ============================================================================
    
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
                'is_running': True  
            }
            
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(coder_state, f)
                
        except Exception as e:
            self.logger.warning(f"Warning: Failed to save coder state: {e}")
    
    def _add_message(self, role: str, content: str):
        """Add message to conversation history and save state."""
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

    # ============================================================================
    # ASYNC EXECUTION AND MONITORING
    # ============================================================================
    
    def _start_output_monitor(self):
        """Start monitoring real-time output from Jupyter session log file."""
        if self._output_monitor_thread and self._output_monitor_thread.is_alive():
            return
            
        def monitor_output():
            last_modified = 0
            last_content_hash = ""
            
            while self._execution_active:
                try:
                    # Monitor the Jupyter session's log file
                    if self._current_log_file and self._current_log_file.exists():
                        # Check file modification time
                        current_modified = self._current_log_file.stat().st_mtime
                        
                        # Read and check content for changes
                        try:
                            with open(self._current_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                full_content = f.read()
                            
                            # Use content hash to detect actual changes
                            import hashlib
                            content_hash = hashlib.md5(full_content.encode()).hexdigest()
                            
                            if content_hash != last_content_hash or current_modified > last_modified:
                                with self._execution_lock:
                                    # Update output lines with log file content
                                    self.output_lines = full_content.split('\n') if full_content else ["Jupyter execution starting..."]
                                    self._save_state()
                                    self.logger.info(f"🔄 Jupyter output updated: {len(self.output_lines)} lines")
                                
                                last_content_hash = content_hash
                                last_modified = current_modified
                        except Exception as read_error:
                            self.logger.error(f"Error reading Jupyter log file: {read_error}")
                    else:
                        # Log file doesn't exist yet, update with placeholder
                        with self._execution_lock:
                            self.output_lines = ["Jupyter session starting, waiting for output..."]
                            self._save_state()
                    
                    time.sleep(10)  # Check every 10 seconds for Jupyter
                except Exception as e:
                    self.logger.error(f"Error in Jupyter output monitor: {e}")
                    break
                    
        self._output_monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        self._output_monitor_thread.start()
    
    def _stop_output_monitor(self):
        """Stop the output monitor thread."""
        self._execution_active = False
        if self._output_monitor_thread and self._output_monitor_thread.is_alive():
            self._output_monitor_thread.join(timeout=5)
    
    def _execute_async(self, code: str, goal: str):
        """Execute Jupyter cell asynchronously with real-time monitoring."""
        def execute_in_background():
            try:
                with self._execution_lock:
                    self._execution_active = True
                    self._execution_result = None
                
                result = self._execute_with_monitoring(code, goal)
                
                with self._execution_lock:
                    self._execution_result = result
                    if result and result.output:
                        self.output_lines = result.output.split('\n')
                    else:
                        self.output_lines = ["Jupyter execution completed"]
                    self._save_state()
                    
            except Exception as e:
                with self._execution_lock:
                    self.output_lines = [f"Jupyter execution error: {str(e)}"]
                    self._save_state()
            finally:
                self._stop_output_monitor()
                
        self._execution_thread = threading.Thread(target=execute_in_background, daemon=True)
        self._execution_thread.start()
    
    def _execute_with_monitoring(self, code: str, goal: str):
        """Execute Jupyter cell with real-time monitoring using log file."""
        try:
            # Get the log file path from Jupyter session
            with self._execution_lock:
                self._current_log_file = self.jupyter_session.log_file_path
            
            # Start monitoring the log file
            self._start_output_monitor()
            
            self.logger.info(f"🔄 Starting Jupyter execution with log file: {self._current_log_file}")
            
            # Execute using the Jupyter session
            result = self.jupyter_session.append_cell(code, goal)
            
            return result
        except Exception as e:
            self.logger.error(f"Jupyter execution error: {e}")
            return None
    
    def _wait_for_execution(self, timeout=None):
        """Wait for Jupyter execution to complete and return the result."""
        if self._execution_thread:
            self._execution_thread.join(timeout=timeout)
            
        with self._execution_lock:
            return self._execution_result

    # ============================================================================
    # LLM INTERACTION AND PROMPT GENERATION
    # ============================================================================
    
    def _post_initial_message(self):
        """Generate and post the initial system message to start the conversation."""
        data_preview = generate((self.cfg.agent_workspace_dir.parent), include_file_details=False)
        prompt = f"""
You're an expert Kaggle competitor tasked with implementing a pipeline into a Jupyter notebook. You can modify the details (training parameters, feature engineering, model selection, etc. ), but do not change overall architecture of this pipeline. The goal is to **obtain best score** on this competition.

<task_desc>\n{self.cfg.competition_task_desc}\n</task_desc>

<pipeline>\n{self.draft.description}\n</pipeline>

This is the code abstract of the pipeline:
<code>\n{self.draft.code}\n</code>

Follow the pipeline description and the code abstract to implement it. All the input files are visible in ../input folder, this folder typically contains the competition data and external resouces, including public datasets, models and outputs of other kernels. DO NOT USE /kaggle/input paths in your code. USE ../input instead.

file structure:
    - input/ (../input)
        - competition_id/ # the official competition dataset
        - alice/dataset1/ # other public datasets
        - alice/kernel1/  # referenced kernels
    - working/ 
        - agent.ipynb # the notebook you will be working on (./agent.ipynb)
        - other files

 Here is an abstract of the file structure:

<data_preview>\n{data_preview}\n</data_preview>
"""
        if self.draft.codebase_content is not None:
            prompt += f"""
You will develop the pipeline based on this codebase. Any output files of the codebase, such as csvs, checkpoints, etc., are visible in ./, which is also your current working directory. 

<code>\n{self.draft.codebase_content}\n</code>

You should note that checkpoints generated by this codebase is store in ./ other than ../input. You must load the checkpoint file under the ./ directory for ensemble prediction.

Your code must produce a submission at ./{self.submission_name}, this is EXTREMELY IMPORTANT. Before generating the submission, you must print the value of the evaluation metric computed on a hold-out validation set. You can use custom evaluation functions during training, but the final metric **MUST FOLLOW THE EVALUATION SECTION IN THE TASK DESCRIPTION** on a validation set. If other kernels with {self.submission_name} are provided in the input folder, you can ensemble them before generating your own submission. This is important because we will pick your best code based on this metric. You are allowed to load the checkpoints of other models. Do not contain any absolute paths in your code. Time limit per run is 2 hours. Your code will be killed if timeout. 

Your code will be executed on a single A6000 GPU. Use large batchsizes to maximize the gpu utilization. If the code segment is provided in this prompt, you should follow the input/output structure. You are allowed to install any packages you need or inspect the workspace (e.g., print file contents, check folder structure). Always use gpu for acceleration. DO NOT USE ABSOLUTE PATHS IN YOUR CODE.

The workspace will be maintained across iterations. That is, if your first iteration code produces a checkpoint, you can load it in the second iteration. You can ensemble submissions generated by yourself and other kernels. You should generate model checkpoints for future loading. If you load the external submissions successfully but failed to merge them with your own predictions, you should print the headers of the external submission and your own predictions and check if the ids are aligned. All the external submissions are valid. Your predictions should be in the same format as them.
"""
        if self.evaluator:
            prompt += "To evaluate your submission locally. You should also generate a submission file on the validation set. All the validation data are typically structured similarly to the test data. An external grader will be used to evaluate your validation submission. That is to say, you should generate TWO submission files: one is for the validation set and the other is for the test set. Generate two submission files in the same code cell."

        prompt += f"""
We have installed the following packages: {", ".join(self._get_packages())}. You are allowed to install any packages by running `pip install <package_name>` in your script. Your installation will take effect in the NEXT cell.

A persistent Jupyter Notebook session is maintained. Your proposed code cell will be directly appended to the notebook and executed. You should separate data loading, training and evaluation in different cells. Now, please propose THE FIRST CELL of your code (not your full code!) using the following format:

<goal>
The explanation of your first cell. You should describe the desired execution time and output of this cell. Explain how to interpret the execution output. 
</goal>

<code>
The content of this cell. Do not wrap the code in a markdown block. Your code will be appended to the notebook, which is stored at ./agent.ipynb. Your code must print necessary information after each milestone.
</code>

You should propose exactly ONE code cell per iteration and I'll tell you the output of your code cell. 
"""
        if self.evaluator:
            prompt += """
<validation_submission>
The name of the submission file for the validation set. e.g. validate_submission.csv. If your current code cell does not produce two submission files, leave this as None. 
</validation_submission>

<submission>
The name of the submission file for the test set. e.g. submission.csv. This submission should be ready for Kaggle submission. If your current code cell does not produce two submission files, leave this as None.
</submission>

The validation_submission tag and the submission tag should must be both empty or both non-empty.
"""
        else:
            prompt += f"""
<metric>
The value of the **final metric** (not the training loss value) if the code execution was successful, producing {self.submission_name} and the metric is calculated properly on the full test set (not dummy or partial). Otherwise, if the code execution was terminated or output messages indicate any training/execution failure, leave it as None. This section should be a real number or None. Report decimal number if the metric is a float.
</metric>
"""
        self.llm.add_message(role="system", content=prompt)
        self._add_message("agent", prompt)
    
    def _get_feedback_prompt(self, result: ExecutionResult) -> str:
        """Generate feedback prompt based on execution result."""
        prompt = f"The execution takes {result.execution_time:.2f} seconds and ends with the following output:\n"
        prompt += f"<output>\n{self._truncate_result_output(result.output)}\n</output>"
        
        if not result.success:
            if result.timeout:
                prompt += "Execution timed out and was killed by system.\n"
            elif result.llm_terminated:
                prompt += f"Execution was terminated by LLM with reason: {result.error}\n"
            else:
                prompt += f"Execution terminated unexpectedly. Traceback: {result.error}\n"
            prompt += "Please investigate the issue and propose a fix. Your next code cell should address the problem identified in the previous cell. Consistent with Jupyter Notebook, all current temporary variables have been loaded into memory.\n"
        else:
            prompt += "Execution completed successfully. You should keep updating your code (e.g., try different hyperparameters, augmentations, model architectures) after you have made successful submission. Your best submission will be recorded.\n"

        return prompt

    # ============================================================================
    # RESULT PROCESSING AND REPORT GENERATION
    # ============================================================================
    
    def _truncate_result_output(self, output: str) -> str:
        """Truncate long output for better readability."""
        output_lines = output.splitlines()
        if len(output_lines) <= 80:
            return output 
        
        # Try to omit long sentences 
        filtered_lines = [line for line in output_lines if len(line.split()) < 100]

        if len(filtered_lines) <= 80:
            return "\n".join(filtered_lines)

        # If still too long, truncate the output
        return "\n".join(filtered_lines[:40] + ["... (truncated) ..."] + filtered_lines[-40:])
    
    def _process_metric_report(self, response):
        """Process metric report from LLM response and update best metrics."""
        if "none" in response['metric'][0].lower():
            return
        
        metric = MetricValue(float(response['metric'][0]), maximize = not self.is_lower_better)
        self.logger.info(f"Captured metric: {metric}")
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_code = deepcopy(self.jupyter_session.cells)
            shutil.copyfile(
                self.cfg.agent_workspace_dir / self.submission_name,
                self.best_submission_path
            )
            self.metric_updater.post(metric, self.best_submission_path)
    
    def _generate_report(self) -> Pipeline:
        """Generate final report and pipeline summary."""
        prompt = f"""
{self.last_feedback}

Please summarize the full code that achieved the best metric value. Explain your code with the best score and the modifications you made to the codebase (if the codebase is provided).

You should respond in the following format:
"""
        if not self.evaluator:
            prompt += """
<metric>If your last code cell produced a submission file, include the evaluation metric here. Do not provide any other information in this tag. If no submission was produced, leave this as None.</metric>
"""
        prompt += """
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
        
        if self.evaluator:
            report = self.llm.query(required_fields=["description", "code", "suggestions"], check_fn = lambda x: len(x["suggestions"]) == 1)
        else:
            report = self.llm.query(required_fields=["metric", "description", "code", "suggestions"], check_fn = lambda x: len(x["suggestions"]) == 1)
            self._process_metric_report(report)

        self._add_message("llm", report['_raw_content'])
        submission = self.cfg.agent_workspace_dir / self.submission_name
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
                'is_running': False, 
                'completed': True
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(coder_state, f)
            self.logger.info(f"🏁 Coder {self.draft.id} completed with final best metric: {self.best_metric}")
        except Exception as e:
            self.logger.warning(f"Warning: Failed to save final coder state: {e}")
        
        full_code = self.best_code.get_notebook_code() if self.best_code is not None else "Not available"
        self.logger.info(f"Full code: {full_code}")

        return Pipeline(
            id=self.draft.id,
            title=self.draft.title,
            description=report["description"][0],
            code=report["code"][0],
            full_code=full_code,
            referenced_private_data=False,
            metric=self.best_metric,
            submission=submission,
            output_dir=self.cfg.agent_workspace_dir,
            suggestions=report["suggestions"][0],
            datasets=self.draft.datasets,
        )

    # ============================================================================
    # MAIN EXECUTION FLOW
    # ============================================================================

    def run(self) -> Pipeline:
        """Main execution flow for the code agent."""
        self._post_initial_message()
        
        for iteration in range(self.cfg.agent_num_iterations_code_agents):
            self.logger.info(f"Iteration {iteration}")
            self.iteration = iteration
            
            # Check time limit
            time_elapsed = time.time() - self.start_time
            if iteration + 1 == self.cfg.agent_num_iterations_code_agents \
                or time_elapsed > self.cfg.agent_time_limit_code_agents:
                self.llm.pop_message()
                break 

            # Get LLM response
            if self.evaluator:
                def check_fn(response: dict) -> bool:
                    if len(response["suggestions"]) != 1:
                        return False 
                    a = "none" in response["validation_submission"][0].lower()
                    b = "none" in response["submission"][0].lower()
                    return a == b
                response = self.llm.query(required_fields=["code", "goal", "validation_submission", "submission"], check_fn=check_fn)
            else:
                response = self.llm.query(required_fields=["metric", "code", "goal"])
                if iteration:
                    response = self.llm.query(required_fields=["metric", "code", "goal"])
                    self._process_metric_report(response)
                else:
                    response = self.llm.query(required_fields=["code", "goal"])

            # Extract code and goal, update state
            code, goal = response['code'][0], response['goal'][0]
            self._add_message("llm", response['_raw_content'])
            self._update_code(code)

            self.logger.info(f"Executing code with monitoring: {code}")
            # Execute code with monitoring
            self._execute_async(code, goal)
            
            result = self._wait_for_execution()
            self.logger.info(f"output: {result.output}")
            self.logger.info(f"error: {result.error}")
            self.logger.info(f"success: {result.success}")
            self.logger.info(f"execution_time: {result.execution_time}")
            
            # Final output update
            if result and result.output:
                output_lines = result.output.split('\n')
                self._update_output(output_lines)

            # Generate feedback and continue conversation
            feedback_prompt = self._get_feedback_prompt(result)
            
            if self.evaluator:
                if result.success and "none" not in response["submission"][0].lower():
                    validate_submission = self.cfg.agent_workspace_dir / response["validation_submission"][0]
                    submission = self.cfg.agent_workspace_dir / response["submission"][0]
                
                    if validate_submission.exists():
                        metric = self.evaluator.evaluate(validate_submission)
                        if metric.success:
                            score = MetricValue(metric.score, maximize=not self.is_lower_better)
                            feedback_prompt += f"\nThe evaluation metric on the validation set is {score}.\n"
                            if not submission.exists():
                                feedback_prompt += f"\nYour code failed to generate a submission file on the test set.\n"
                            elif score > self.best_metric:
                                self.best_metric = score
                                self.best_code = deepcopy(self.jupyter_session.cells)
                                shutil.copyfile(
                                    submission,
                                    self.best_submission_path
                                )
                                self.metric_updater.post(score, self.best_submission_path)
                        else:
                            feedback_prompt += f"\nYour code failed to generate a valid submission file on the validation set. Error message: {metric.message}\n"
                    else:
                        feedback_prompt += f"\nYour code failed to generate a submission file on the validation set.\n"


            self.last_feedback = feedback_prompt
            feedback_prompt += """
Now, respond in the following format:
"""
            if self.evaluator:
                feedback_prompt += """
<validation_submission>
The name of the submission file for the validation set. e.g. validate_submission.csv. If your current code cell does not produce a submission file on the validation set, leave this as None.
</validation_submission>

<submission>
The name of the submission file for the test set. e.g. submission.csv. This submission should be ready for Kaggle submission. If your current code cell does not produce a submission file on the test set, leave this as None.
</submission>

The validation_submission tag and the submission tag should must be both empty or both non-empty.
"""
            else:
                feedback_prompt += """
<metric>The evaluation metric of your submission. Only include this part if your last code cell produced a full, successful submission file. Otherwise, leave this as None.</metric>
"""
            feedback_prompt += """
<goal>Describe the goal and how to inspect the output of your next code cell</goal>

<code>
The content of your next code cell. Following the previous format, do not wrap your code within markdown code marks. You should keep updating your code (e.g., try different hyperparameters, augmentations, model architectures) even after you have made successful submission. Always evaluate your submission and print the metric on a validation set.
</code>
"""

            self.llm.add_message(role="user", content=feedback_prompt)
            self._add_message("agent", feedback_prompt)

        # Finalize and generate report
        if self.best_submission_path.exists():
            shutil.copy(self.best_submission_path, self.cfg.agent_workspace_dir / self.submission_name)
        
        result = self._generate_report()
        self.jupyter_session.shutdown()
        del self.jupyter_session

        return result