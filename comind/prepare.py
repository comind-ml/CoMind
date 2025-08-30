from comind.config import Config
from comind.community import Pipeline
from comind.agent import MetricUpdater
from comind.utils import get_logger, Conversation, generate
from comind.utils.jupyter_session import JupyterSession, ExecutionResult
from comind.utils.metric import WorstMetricValue, MetricValue

class PrepareAgent:
    def __init__(
        self, 
        cfg: Config, 
        pipeline: Pipeline, 
        is_lower_better: bool, 
        metric_updater: MetricUpdater,
        env_name: str
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.metric_updater = metric_updater
        self.is_lower_better = is_lower_better

        self.logger = get_logger(f"prepare-{pipeline.id}", self.cfg.agent_workspace_dir / "prepare.log")
        self.llm_logger = get_logger(f"llm-{pipeline.id}", self.cfg.agent_workspace_dir / "llm.log", file_only=True)
        self.logger.info(f"Prepare {pipeline.id} using conda environment {env_name}")

        self.llm = Conversation(cfg.llm, logger=self.llm_logger)
        self.jupyter_session = JupyterSession(cfg, env_name)
    
    def _post_initial_message(self):
        data_preview = generate((self.cfg.agent_workspace_dir.parent / "input"))
        prompt = f"""
You are a professional Kaggle competitor and tasked with preparing the environment for a script. Your goal is to setup the enrionment and correct minor issues (e.g., incorrect paths, etc.) in the code, until the loss curve is stable and the final submission file is generated at ./submission.csv. 
<task_desc>\n{self.cfg.competition_task_desc}\n</task_desc>

<code>\n{self.pipeline.full_code}\n</code>

This code runs smoothly on kaggle's default settings. However, different from kaggle's default settings, all the input files are visible in ../input folder, this folder contains the competition data and external resouces, including public datasets, models and outputs of other kernels. DO NOT USE /kaggle/input paths in your code. USE ../input instead. Here is an abstract of the input files (../input):

<data_preview>\n{data_preview}\n</data_preview>

You are allowed to install any packages by running `pip install <package_name>` in your script. A persistent Jupyter Notebook session is maintained. 

You should decompose the full code into several code cells and I will execute them sequentially. Your proposed code cell will be directly appended to the notebook and executed. You should separate data loading, training and evaluation in different cells. DO NOT change the pipeline's code structure and any implementation details, including hyperparameters, model architectures, data augmentations, etc. Just correct the minor issues.

You should correct the code until the loss curve is stable and you have captured the final evaluation metric.

You MUST print out the final evaluation metric before generating the submission file. If the code does not print out the metric, **it is your responsibility to compute the metric and print it out**.

Now, please propose THE FIRST CELL (not your full code!) using the following format:

<goal>
Describe the goal of this cell and the expected output. Mention any changes you made to the code and how to inspect the results.  
</goal>

<code>
The content of this cell. Do not wrap the code in a markdown block. Your code will be appended to the notebook, which is stored at ./agent.ipynb. 
</code>
"""
        self.llm.add_message(role="system", content=prompt)


    def _truncate_result_output(self, output: str) -> str:
        """Truncate long output for better readability."""
        output_lines = output.splitlines()
        if len(output_lines) <= 80:
            return output
        return "\n".join(output_lines[:80]) + "\n..."


    def _get_feedback_prompt(self, result: ExecutionResult) -> str:
        """Generate feedback prompt based on execution result."""
        prompt = f"The execution takes {result.execution_time:.2f} seconds and ends with the following output:\n"
        prompt += f"```\n{self._truncate_result_output(result.output)}\n```"
        
        if not result.success:
            if result.timeout:
                prompt += "Execution timed out and was killed by system.\n"
            elif result.llm_terminated:
                prompt += f"Execution was terminated by LLM with reason: {result.error}\n"
            else:
                prompt += f"Execution terminated unexpectedly. Traceback: {result.error}\n"
            prompt += "Please investigate the issue and propose a fix. Your next code cell should address the problem identified in the previous cell. Consistent with Jupyter Notebook, all current temporary variables have been loaded into memory.\n"
        else:
            prompt += "Execution completed successfully. If submission file was generated, you should report the evaluation metric in the goal tag.\n"

        return prompt


    def run(self) -> Pipeline:
        self._post_initial_message()
        success = False
        metric = WorstMetricValue()

        for iteration in range(self.cfg.agent_num_iterations_code_agents):
            self.logger.info(f"Iteration {iteration}")
            def check_fn(response: dict) -> bool:
                if len(response["code"][0]) < 10:
                    try:
                        _ = float(response["goal"][0])
                        return "none" in response["code"][0].lower()
                    except:
                        return False
                return True

            response = self.llm.query(required_fields=["goal", "code"], check_fn=check_fn)
            goal, code = response["goal"][0], response["code"][0]
            
            if len(code) < 10:
                success = True
                metric = MetricValue(float(goal), maximize = not self.is_lower_better)
                break

            self.logger.info(code)
            result = self.jupyter_session.append_cell(code, goal)
            self.logger.info(f"output: {result.output}")
            self.logger.info(f"error: {result.error}")
            self.logger.info(f"success: {result.success}")
            self.logger.info(f"execution_time: {result.execution_time}")

            feedback = self._get_feedback_prompt(result)
            feedback += f"""
Your responses should always contain <goal> (even if you are reporting the metric) and <code> tags. Respond in the following format:

<goal>
Describe the goal of this cell and the expected output. Mention any changes you made to the code and how to inspect the results. If all the issues are resolved and you have captured the final evaluation metric, report the metric here instead of the goal. Report decimal number if the metric is a float. Do not change the name of the tag even if you are reporting the metric.
</goal>

<code>
The content of the next cell. Do not wrap the code in a markdown block. Your code will be appended to the notebook. If all the issues are resolved, submission file is generated at ./submission.csv and you have captured the final evaluation metric, leave this as None. e.g. <code>None</code>
</code>
"""

            self.llm.add_message(role="user", content=feedback)
        
        self.logger.info(f"Success: {success}, metric: {metric}")
        if not success:
            return None 
        
        full_code = self.jupyter_session.get_notebook_code()
        self.logger.info(f"full_code: {full_code}")
        self.jupyter_session.shutdown()
        del self.jupyter_session

        submission_path = self.cfg.agent_workspace_dir / "submission.csv"
        self.metric_updater.post(metric, submission_path)

        return Pipeline(
            id=self.pipeline.id,
            title=self.pipeline.title,
            description=self.pipeline.description,
            code=self.pipeline.code,
            full_code=full_code,
            referenced_private_data=False,
            metric=metric,
            submission=submission_path,
            output_dir=self.cfg.agent_workspace_dir,
            suggestions=self.pipeline.suggestions,
            datasets=self.pipeline.datasets,
        )

