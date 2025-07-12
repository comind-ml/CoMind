import backend
import random
import logging
import requests
import asyncio
import json
import shutil
import time

from utils.config import Config, load_cfg, prep_agent_workspace, load_task_desc
from retriever import Retriever
from executor import Executor
from typing import List, cast
from pathlib import Path
from uuid import uuid4
from omegaconf import OmegaConf
from backend import Conversation, FunctionSpec, query

from utils.response import extract_code, extract_text_up_to_code, wrap_code, wrap_docs, trim_long_string
from utils.metric import MetricValue, WorstMetricValue
from utils.data_preview import generate
from utils import get_timestamp

review_function_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (4-5 sentences) describing the empirical findings. "
                "Examine wheter its goals are achieved. "
                "Summarize the output and mention if the submission.csv was properly produced. "
                "DO NOT suggest fixes or improvements.",
            },
            "output_abs": {
                "type": "string",
                "description": "select representative segments of the output log and mark the remainder as ellipses."
            },
            "metric": {
                "type": "number",
                "description": "If the code **ran successfully and produced submission.csv on full test set (i.e. not dummy or partial)**, report the value of the **final** validation metric. Otherwise, leave it null."
            },
            "is_lower_better": {
                "type": "boolean", 
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy)."
            }
        },
        "required": [
            "is_bug",
            "summary",
            "output_abs",
            "metric",
            "is_lower_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

report_function_spec = FunctionSpec(
    name="submit_report",
    json_schema={
        "type": "object",
        "properties": {
            "pipeline": {
                "type": "string",
                "description": "A detailed description of the pipeline that generated the best results. All hyperparameters, training settings, model architectures, feature engineering, validation metric, and any other relevant information should be included. Describe potential improvements and future work. ",
            },
            "summary": {
                "type": "string",
                "description": (
                    "A comprehensive evaluation of each individual component of the pipeline. For each component, summarize in the following format: \n"
                    "- The component name\n"
                    "    Novelty: 0-10 (0: trivial, 10: clearly novel - major differences from existing well-known methods)\n"
                    "    Your Rationale \n\n"
                    "    Feasibility: 0-10 (0: almost impossible to implement and require extensive engineering, 10: Easy to implement)\n"
                    "    Your Rationale \n\n"
                    "    Effectiveness: 0-10 (0: minimal performance improvement, 10: very strong performance, significantly outperform most baselines)\n"
                    "    Your Rationale \n\n"
                    "    Efficiency: 0-10 (0: very slow, over-dependent on CPU and hard to produce meaningful results within the time limit, 10: high utilization of GPU)\n"
                    "    Your Rationale \n\n"
                    "    Confidence: 0-10 (0: no emprical results, not sure whether the evaluation is correct, 10: fully verified on large scale with abundant results)\n"
                )
            },
            "suggestions": {
                "type": "string",
                "description": (
                    "An extremely detailed description of the weaknesses of the pipeline you found during the implementation. Determine the bottleneck of the pipeline and suggest possible improvements. "
                )
            }
        },
        "required": [
            "pipeline",
            "summary",
            "suggestions",
        ],
    },
    description="Submit a comprehensive report evaluating the whole pipeline.",
)

class CodeAgent:
    def __init__(
        self, 
        task_desc: str, 
        cfg: Config, 
        global_start_time: float,
        global_metric,
        global_lock,
        status_dict = None,
        agent_id: int = 0,
    ):
        self.task_desc    = task_desc
        self.agent_id     = agent_id
        self.cfg          = cfg
        self.metric       = WorstMetricValue()
        self.acfg         = cfg.agent
        self.status_dict  = status_dict
        self.error        = False
        self.best_code    = None

        self.submision_history_dir = cfg.log_dir / "best_submission_history"
        self.submision_history_dir.mkdir(parents=True, exist_ok=True)

        self.code_history_dir = cfg.log_dir / "best_solution_history"
        self.code_history_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor     = Executor(cfg.workspace_dir, timeout=cfg.exec.timeout, use_pty=True)
        self.conversation = Conversation(model=cfg.agent.code.model, temperature=cfg.agent.code.temp)

        self.global_start_time = global_start_time
        self.global_metric = global_metric
        self.global_lock   = global_lock

        self.initialize_logger()
    
    def initialize_logger(self):
        log_format = "[%(asctime)s] %(levelname)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper()), format=log_format, handlers=[]
        )

        logger = logging.getLogger(f"graphml-code-agent-{self.agent_id}")
        log_dir = self.cfg.log_dir / f"coder-{self.agent_id}"

        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = get_timestamp()
        file_handler = logging.FileHandler(log_dir / f"{timestamp}.log")
        file_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(file_handler)

        self.logger = logger

    def initialize(self):
        prompt = {
            "Introduction": "You're an expert Kaggle competitor tasked with implementing a pipeline into Python code. You can modify the details (training parameters, feature engineering, model selection, etc. ), but do not change overall architecture of this pipeline. The goal is to **obtain best score** on this competition. ",
            "Task Description": self.task_desc,
            "Pipeline": self.pipeline,
            "Data Overview": self.data_overview,
            "Instructions": {
                "Response Format": (
                    "- Objective of this implementation and suggestions for output evaluation\n"
                    "- Key technical considerations\n"
                    "- Expected running time (you should ensure that the code will finish within 1 hour)\n"
                    "```python\n"
                    "Your code here\n"
                    "```\n"
                ),
                "Reminders": (
                    "- Read the pipeline and task description carefully.\n"
                    "- Avoid using progress bars in your code.\n"
                    f"- **YOUR CODE MUST PRODUCE SUBMISSON AT ./working{self.agent_id}/submission.csv, THIS IS EXTREMELY IMPORTANT**\n"
                    "- There is one A6000 gpu available for you, **maximize your use of computing resources**. You can use large batchsizes.\n"
                    "- All the provided input data is **stored in './input' directory**.\n"
                    f"- You can use the './working{self.agent_id}' directory to store any temporary files that your code needs to create.\n"
                    "- Include at least one comment explaining your code. **NO PARTS OF THE CODE SHOULD BE SKIPPED OR OMITTED**, don't terminate before finishing the script. Even if your proposed code is a minor change, don't omit any sections that overlap with the previous code.\n"
                    "- Remember, your ultimate goal is to **Obtain best score on this competition**. \n"
                    "- Your code should **print the value of the evaluation metric computed on a hold-out validation set.**\n"
                    "- You can use custom evaluation functions during training, but the final metric **MUST FOLLOW THE EVALUATION SECTION IN THE TASK DESCRIPTION** on a validation set. This is important because we will pick your best code based on this metric.\n"
                    "- We suggest you to test your code at a small scale and print necessary information before utilizing full dataset to get familiar with the data structure and avoid potential format errors. \n"
                    "- Time limit per run is 3 hours. Your code will be killed if timeout."
                    "- Begin by summarizing your understanding of the task, and then propuse your first code.\n"
                )
            }
        }

        prompt["Instructions"] |= self._prompt_environment

        self.logger.info(f"Init prompt: {json.dumps(prompt, indent=2)}")
        self.conversation.add_message(system_message=prompt)
    
    @property
    def _prompt_environment(self):
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
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt
    
    def update_global_metric(self, metric: MetricValue, code: str, submission_path: Path):
        with self.global_lock:
            global_metric = MetricValue(value=self.global_metric["value"], maximize=self.global_metric["maximize"])
            if metric > global_metric:
                self.logger.info(f"New global best submission found. ")
                self.global_metric["value"] = metric.value
                self.global_metric["maximize"] = metric.maximize

                time_since_start = time.time() - self.global_start_time

                shutil.copy(
                    submission_path, self.submision_history_dir / f"submission_{time_since_start}.csv"
                )

                with open(self.code_history_dir / f"solution_{time_since_start}.py", "w") as f:
                    f.write(code)

    def set_status(self, status: str):
        if self.status_dict is not None:
            self.status_dict[self.agent_id] = f"Step {self.step}: {status}"

    def parse_exec_result(self, code_desc: str, code: str, exec_result) -> str:

        output = exec_result['output']
        output += f"\n\n[System] Execution time: {exec_result['execution_time']:.2f} seconds.\n"
        self.logger.info(f"Execution time: {exec_result['execution_time']:.2f} seconds.")
        if exec_result['error'] == "Timeout":
            self.logger.info("Execution timed out.")
            output += "[System] Process was killed due to timeout.\n"

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
            "Include essential information about the result, including warnings, errors, and the final metric. "
        )
        
        prompt = {
            "Introduction": introduction,
            "Implementation": wrap_code(code),
            "Goals and explanation": code_desc,
            "Execution output": wrap_code(trim_long_string(output, threshold=11000, k=5000), lang=""),
        }

        while True:
            try:
                response = cast(
                    dict,
                    query(
                        system_message=prompt,
                        user_message=None,
                        func_spec=review_function_spec,
                        model=self.acfg.feedback.model,
                        temperature=self.acfg.feedback.temp,
                    ),
                )

                result = (
                    f"Terminal output (truncated): \n```\n{response['output_abs']}\n```\n"
                    f"Execution summary: \n{response['summary']}\n"
                    f"Execution time: {exec_result['execution_time']:.2f} seconds.\n"
                )

                metric = response["metric"]
                submission_path = self.cfg.workspace_dir / f"working{self.agent_id}" / "submission.csv"

                if not isinstance(metric, (int, float)) or not submission_path.exists():
                    metric = WorstMetricValue()
                else:
                    metric = MetricValue(
                        response['metric'], maximize=not response['is_lower_better']
                    )
                
                break

            except Exception as e:
                self.logger.error(f"Error in parsing execution result: {e}")
                self.logger.info("Retrying...")
                continue
        
        self.logger.info(f"Metric: {metric}")
        
        if metric > self.metric and submission_path.exists():
            self.logger.info(f"New best submission found.")

            self.metric = metric 
            self.best_code = code
            best_submission_dir = self.cfg.workspace_dir / f"best_submisson{self.agent_id}"
            best_submission_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(submission_path, best_submission_dir / "submission.csv")

            self.update_global_metric(
                metric=metric,
                code=code,
                submission_path=submission_path,
            )
            
        return result

    def _step(self, is_final_step: bool = False):
        self.set_status("Querying the agent...")

        if is_final_step:
            self.conversation.pop_message()
            self.conversation.add_message(user_message="Please summarize the results and submit a comprehensive report.")

            report = cast(
                dict,
                self.conversation.query(func_spec=report_function_spec)
            )

            self.report = report
            report_text = json.dumps(report, indent=2)
            self.logger.info(f"Report: {report_text}")

            return

        get_valid_response = False
        for _ in range(4):
            response = self.conversation.query()

            code = extract_code(response)
            code_desc = extract_text_up_to_code(response)

            if len(code) < 10 or code is None:
                self.logger.info("Code extraction failed. Prompting agent to retry.")
                self.conversation.pop_message()
                continue

            get_valid_response = True

            self.set_status("Running code...")
            self.logger.info(f"Running code: \n{code}\n")
            result = asyncio.run(self.executor.run(code, agent_file_name=f"agent_{self.agent_id}.py"))
            self.logger.info(f"Agent finished running the code with output \n{trim_long_string(result['output'])}\n.")

            self.set_status("Analyzing the execution result...")
            summary = self.parse_exec_result(code_desc, code, result)

            remaining_steps = self.acfg.max_steps - self.step - 1
            remaining_time = self.ddl - time.time()

            self.logger.info(f"Execution summary: {summary}")
            self.conversation.add_message(user_message=(
                f"Remaining steps: {remaining_steps}; Remaining time: {remaining_time} seconds\n"
                f"I ran your code and summarized the execution result:\n"
                f"{summary}\n"
                f"Now, please choose your next action and propose code using the same response format as before. Remember, output a self-contained code, no part of it should be omitted. Keep the final validation metric same as the metric mentioned in task description.\n"
                "A) Fix runtime errors (if any)\n"
                "B) Do hyperparameter tuning\n"
                "C) Include ideas that were not implemented yet\n"
                "D) Add possible improvements\n"
                "E) Run on a larger scale (moderately increase training epochs, etc.). You should refer to the previous execution time we reported. Remember your code will be killed if timeout.\n"
            ))

            break
        
        if not get_valid_response:
            self.set_status("Failed to get a valid response. Aborting.")
            self.error = True
        
    def run(self, pipeline: str) -> dict:
        self.pipeline = pipeline

        self.data_overview = generate(self.cfg.data_dir, include_file_details=True, simple=False)
        self.initialize()

        self.start_time = time.time()
        self.ddl = time.time() + self.acfg.timeout

        for step in range(self.acfg.max_steps):
            self.step = step

            is_final_step = step == self.acfg.max_steps - 1 or time.time() > self.ddl

            self._step(is_final_step)

            if self.error or is_final_step:
                break
        
        if self.error:
            return None

        self.set_status("Finished.")

        submission_path = None if self.metric == WorstMetricValue() else (
            self.cfg.workspace_dir / f"best_submisson{self.agent_id}" / "submission.csv"
        )
    
        self.report |= {
            "metric": self.metric,
            "code": self.best_code,
        }
        
        return {
            "submission": submission_path,
            "report": self.report,
        }
        

