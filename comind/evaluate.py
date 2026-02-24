from pathlib import Path
from dataclasses import dataclass
import json
import shutil
from comind.config import Config
from comind.utils import generate, Conversation, get_logger
from comind.utils.execute import ExecutionResult, execute, execute_file
from comind.kaggle.leaderboard import get_leaderboard, get_rank

@dataclass
class EvaluationResult:
    score: float | None
    rank: float | None
    success: bool
    message: str

class Evaluator:
    def __init__(self, cfg: Config, is_lower_better: bool):
        self.leaderboard = get_leaderboard(cfg.competition_id)
        self.is_lower_better = is_lower_better
        self.cfg = cfg
        self.llm_cfg = cfg.llm

        self.workspace = cfg.agent_external_data_dir / f"{cfg.competition_id}"
        self.public_dir = self.workspace / "public"
        self.private_dir = self.workspace / "private"
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)

        self.sample_submission_path = self.public_dir / f"validate_sample_{cfg.agent_submission_file_name}"
        self.split_dataset_script_path = self.workspace / "split_dataset.py"
        self.eval_script_path = self.workspace / "evaluate.py"

        self.logger = get_logger("Evaluator", cfg.agent_workspace_dir / "evaluate.log")
        self.llm_logger = get_logger("LLM (Evaluator)", cfg.agent_workspace_dir / "llm.log", file_only=True)

        self.llm = Conversation(self.llm_cfg, self.llm_logger)

    def get_rank(self, score: float) -> float:
        return get_rank(self.leaderboard, score)
    
    def evaluate(self, submission: Path) -> EvaluationResult:
        result = execute_file(self.workspace, self.eval_script_path, ["--public_dir", self.public_dir.absolute(), "--private_dir", self.private_dir.absolute(), "--pred", submission.absolute()])
        assert result.success, result.output + result.error
        with open(self.private_dir / "eval_report.json", "r") as f:
            eval_report = json.load(f)
        
        score = eval_report["score"]
        rank = self.get_rank(score) if score is not None else None
        return EvaluationResult(score, rank, eval_report["success"], eval_report["message"])

    def _get_instruction(self) -> str:
        data_preview = generate(self.cfg.competition_input_dir)

        prompt = f"""
You are an experienced machine learning engineer. Please generate two self-contained Python code for local evaluation of a Kaggle agent. Your code should be robust, reusable, accept command-line arguments and print necessary information. 

## Background

- Kaggle competitions usually provide labels only for the training set. To evaluate an agent locally, we need to split the training set into a training and validation split. 
- The validation set must hide its labels from the agent. The agent only sees the training set (with labels) and the validation inputs (without labels). 
- The hidden validation labels will be stored separately and used only for offline evaluation.
- Importantly: ./public must never contain validation labels. Validation labels are saved only in ./private.

## Kaggle Competition Description

{self.cfg.competition_task_desc}

## Data Preview 

({self.cfg.competition_input_dir.absolute()}/)
{data_preview}

## Deliverables

Please generate two scripts (both in Python 3, runnable from the command line):

**1) split_dataset.py**

**Goal**: Split the original training data into 90% training and 10% validation. Store validation inputs (without labels) in ./public, and validation labels in ./private. The training set (with labels) and original test set must remain in ./public, preserving the original structure as closely as possible. The structure of validation inputs should also match the test set. Generate a sample validate submission validate_sample_{self.cfg.agent_submission_file_name} under ./public. All original data (training and test) are visible in {self.cfg.competition_input_dir.absolute()}.

**Example**: If the original data is structured as:
({self.cfg.competition_input_dir.absolute()}/)
- kaggle_evaluation/ (official evaluation tool provided by Kaggle)
    - __init__.py
    - ...
- train.csv
- train/
- test.csv
- test/
- sample_submission.csv

You should split the dataset into:
(./public/)
- kaggle_evaluation/ (official evaluation tool provided by Kaggle) (unchanged, soft links)
    - __init__.py
    - ...
- train.csv (this contains 90% of the training data)
- train/ (this contains 90% of the training data, keep unchanged data as soft links)
- test.csv (unchanged, soft link)
- test/ (unchanged, soft link)
- sample_submission.csv (unchanged, soft link)
- validate.csv (this contains 10% of the training data with labels withheld)
- validate/ (soft links)
- validate_sample_submission.csv (a sample submission file for validation set)

(./private/)
- validate.csv (labels of validation set)

If the training data contains zip files, you should extract them to the public directory before splitting the dataset. You should always print the directory structure after the split. Do not extract files to the original directory and keep it unchanged.

If the training data contains multiple classes, you should use **stratified sampling**. You should strictly follow the evaluation metric mentioned in the task description and ensure the validation set is representative of the overall class distribution. Never write validation labels into ./public.

Your code will be executed by command line as follows:

```bash
python split_dataset.py --input_dir {self.cfg.competition_input_dir.absolute()} --public_dir ./public --private_dir ./private 
```

DO NOT store the training and test files in other folders such as ./public_<TIMESTAMP>, the ./public folder will be exposed to later code agent. Make sure the ./public directory has similar structure with the original data folder.

**2) evaluate.py**

Goal: Evaluate the agent’s predictions on validation set against the hidden ground truth (./private/...). Output evaluation results (json format) to console and write ./private/eval_report.json.

It will be executed by command line as follows:

```bash
python evaluate.py --public_dir ./public --private_dir ./private --pred <path to the validation submission file>
```

We will pass the path to the sample validation submission file as the argument to your evaluate.py script. It typically produces low scores.

The script should generate in the following json format at ./private/eval_report.json:

{{
    "score": A float number represents the evaluation score on the validation set. Do not omit this field. If the evaluation is unsuccessful or the predictions are invalid, this field should be set to null,
    "success": A boolean value indicates whether the evaluation was successful or not,
    "message": A string provides additional information about the evaluation result. Leave it an empty string if the predictions are valid and evaluation is successful. Otherwise provide necessary details on why it failed.
}}

Do not raise any error or exception. If the evaluation is unsuccessful, you should set the score to null and provide a detailed explanation in the message field.

Now, let's write these two scripts step by step. Your should first generate split_dataset.py. We will execute the code by command line as mentioned above. You should correct the code in case of any issues. You should always generate full, self-contained code. No part of the code should be omitted.

Respond in the following format:

<current_file>
This should be either split_dataset.py or evaluate.py. Leave this as None if both are generated and functioned. This indicates the current file you are editing.
</current_file>

<explanation>
You explanation on the workflow of your code.
</explanation>

<code>
The full content of the current file. Do not wrap the code in a markdown block. Leave this as None if both are generated and functioned.
</code>
"""
        return prompt

    def _truncate_result_output(self, output: str) -> str:
        output_lines = output.splitlines()
        if len(output_lines) <= 80:
            return output
        return "\n".join(output_lines[:40]) + "\n..." + "\n".join(output_lines[-40:])

    def _get_feedback_prompt(self, result: ExecutionResult) -> str:
        prompt = "The execution ends with the following output:\n"
        prompt += f"<output>\n{self._truncate_result_output(result.output)}\n</output>"
        if not result.success:
            prompt += f"Execution terminated unexpectedly. Traceback: {result.error}\n"
        else:
            prompt += "Execution completed successfully."
        
        return prompt

    def setup(self):
        prompt = self._get_instruction()
        self.llm.add_message(role="system", content=prompt)
        while True:
            def check_fn(response: dict) -> bool:
                if not response["current_file"][0] in ["split_dataset.py", "evaluate.py"]:
                    if "none" in response["current_file"][0].lower():
                        return self.split_dataset_script_path.exists() and self.eval_script_path.exists()
                    return False
                return True
            response = self.llm.query(required_fields=["current_file", "code"], check_fn=check_fn)
            current_file = response["current_file"][0]

            if "none" in current_file.lower():
                break

            code = response["code"][0]
            self.logger.info(f"Executing code: {code}")
            args = ["--public_dir", self.public_dir.absolute(), "--private_dir", self.private_dir.absolute()]

            if current_file == "evaluate.py":
                args += ["--pred", self.sample_submission_path.absolute()]
            else:
                args += ["--input_dir", self.cfg.competition_input_dir.absolute()]
                shutil.rmtree(self.public_dir)
                shutil.rmtree(self.private_dir)
                self.public_dir.mkdir(parents=True, exist_ok=True)
                self.private_dir.mkdir(parents=True, exist_ok=True)

            result = execute(self.workspace, code, current_file, args)
            self.logger.info(f"Execution result: {result.output}")
            self.logger.info(f"Execution error: {result.error}")
            self.logger.info(f"Execution success: {result.success}")
            feedback = self._get_feedback_prompt(result)
            feedback += """
Now, respond in the following format:

<current_file>
This should be either split_dataset.py or evaluate.py. **Leave this as None if both are generated and functioned**. This indicates the current file you are editing. You should keep editing the current file until it is fully functional.
</current_file>

<explanation>
You explanation on the workflow of your code.
</explanation>

<code>
The full content of the current file. Do not wrap the code in a markdown block. Leave this as None if both are generated and functioned. You must propose full code even if its a small modification.
</code>

You should make sure your evaluate.py will generate ./private/eval_report.json in the correct format.
"""
            self.llm.add_message(role="user", content=feedback)
