import json
import pickle
import shutil
import queue
import time
import threading
import subprocess
from copy import deepcopy
from rich.status import Status
from datetime import datetime
from comind.config import Config
from pathlib import Path
from comind.community import Pipeline, Dataset, Draft
from comind.utils import query_llm, MetricValue, WorstMetricValue, extract_fields, copytree, extract_archives, get_logger
from comind.evaluate import Evaluator
from comind.kaggle import *

class MetricUpdater:
    def __init__(self, cfg: Config, best_metric: MetricValue, agent=None):
        self.cfg = cfg
        self.q = queue.Queue()
        self._stop = False 
        self.best_metric = best_metric
        self.start_time = cfg.start_time
        self.agent = agent
        self.logger = get_logger("MetricUpdater", cfg.agent_workspace_dir / "metric.log")
        self.logger.info(f"MetricUpdater initialized with best metric: {best_metric}")

    def post(self, metric: MetricValue, submission: Path):
        if not isinstance(metric, WorstMetricValue):
            self.logger.info(f"{metric} posted from path {submission}")
            self.q.put((metric, submission))
    
    def stop(self):
        self._stop = True
        self.q.put((WorstMetricValue(), None))

    def _update_best_metric(self, metric: MetricValue, submission: Path):
        if metric > self.best_metric:
            old_best = self.best_metric
            self.best_metric = metric 
            time_elapsed = time.time() - self.start_time

            shutil.copy(submission, self.cfg.agent_workspace_dir / "submission" / f"submission_{time_elapsed}.csv")

            self.logger.info(f"🏆 Global best metric updated: {old_best} -> {self.best_metric}")

            if self.agent and hasattr(self.agent, 'save_agent_state'):
                try:
                    self.agent.save_agent_state()
                    self.logger.info(f"🏆 Agent state saved with new global best: {self.best_metric}")
                except Exception as e:
                    self.logger.warning(f"Failed to save agent state after metric update: {e}")

    def run(self):
        self._stop = False
        while not self._stop:
            metric, submission = self.q.get()
            if not isinstance(metric, WorstMetricValue):
                self._update_best_metric(metric, submission)

from comind.coder import CodeAgent
from comind.prepare import PrepareAgent

class Agent:
    def __init__(self, cfg: Config, fetch_data_only=False):
        self.cfg                      = cfg
        self.llm_cfg                  = cfg.llm
        self.ideas : list[str]        = []
        self.reports : list[Pipeline] = []
        self.reproduced_reports : list[str] = []
        self.code_agents              = []
        self.datasets: dict[str, Dataset] = {}
        self.task_desc                = cfg.competition_task_desc
        self.submission_name          = cfg.agent_submission_file_name

        (self.cfg.agent_workspace_dir / "submission").mkdir(parents=True, exist_ok=False)
        
        self.logger = get_logger("Agent", cfg.agent_workspace_dir / "agent.log")
        self.llm_logger = get_logger("LLM (Agent)", cfg.agent_workspace_dir / "llm.log", file_only=True)
        
        # Setup conda workspace configuration
        self.conda_envs_dir           = self._setup_conda_workspace(self.cfg.agent_workspace_dir)
        self.cfg.conda_envs_dir       = self.conda_envs_dir

        self.is_lower_better          = self._query_is_lower_better()

        with Status("Fetching external data..."):
            self._fetch_external_data()
        if fetch_data_only:
            return

        with Status("Summarizing kernels..."):
            self._summarize_public_kernels()

        best_metric = WorstMetricValue()
        for report in self.reports: 
            if report.metric > best_metric and report.submission is not None:
                best_metric = report.metric
                shutil.copy(report.submission, self.cfg.agent_workspace_dir / "submission" / self.submission_name)

        self.metric_updater = MetricUpdater(cfg, best_metric, self)
        self.evaluator = None 
        if self.cfg.agent_separate_validation_submission:
            evaluator_cfg = deepcopy(self.cfg)
            evaluator_cfg.agent_workspace_dir = self.cfg.agent_external_data_dir / self.cfg.competition_id
            self.evaluator = Evaluator(evaluator_cfg, self.is_lower_better)
            self.cfg.competition_input_dir = self.evaluator.public_dir

    def save_agent_state(self):
        """Save current agent state for monitoring."""
        state_file = self.cfg.agent_workspace_dir / "agent_state.pkl"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        global_best_metric = str(self.metric_updater.best_metric) if hasattr(self.metric_updater, 'best_metric') else "N/A"
        
        monitor_state = {
            'ideas': self.ideas,
            'reports': self.reports,
            'datasets': self.datasets,
            'code_agents': self.code_agents,
            'cfg': self.cfg,
            'global_best_metric': global_best_metric
        }
        
        with open(state_file, 'wb') as f:
            pickle.dump(monitor_state, f)
        
        return state_file

    def _setup_conda_workspace(self, workspace_dir: Path) -> Path:
        """Setup conda directories for workspace-local environments."""
        # Create conda environments directory
        conda_envs_dir = workspace_dir / "conda_envs"
        conda_envs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Setup conda workspace directory at {conda_envs_dir}")
        return conda_envs_dir
    
    def _clone_conda_environment(self, source_env_name: str, target_env_name: str):
        """Clone a conda environment to workspace-local location using explicit prefix.
        If target already exists, remove it first.
        """
        try:
            # Resolve source and target
            if source_env_name != self.cfg.agent_base_conda_env_name:
                source_env_name = self.conda_envs_dir / source_env_name
            target_env_path = self.conda_envs_dir / target_env_name

            # If target exists, remove it first
            if target_env_path.exists():
                self.logger.info(f"Target env {target_env_path} already exists. Removing...")
                try:
                    subprocess.run(
                        ['conda', 'env', 'remove', '--prefix', str(target_env_path), '-y'],
                        check=True, capture_output=True, text=True
                    )
                    self.logger.info(f"Removed existing env at {target_env_path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to remove existing env: {e}")
                    self.logger.error(f"stdout: {e.stdout}")
                    self.logger.error(f"stderr: {e.stderr}")
                    raise

            # Clone the environment
            self.logger.info(f"Cloning conda environment {source_env_name} -> {target_env_path}")
            subprocess.run(
                ['conda', 'create', '--prefix', str(target_env_path), '--clone', str(source_env_name), '-y'],
                check=True, capture_output=True, text=True
            )

            self.logger.info(f"Successfully cloned conda environment to {target_env_path}")
            return target_env_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone conda environment: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            raise
            
    def _query_is_lower_better(self) -> bool:
        prompt = f"""
You are a Kaggle competitor tasked with achieving the highest possible score on the competition. 

<task_desc>\n{self.task_desc}\n</task_desc>

The first step is to determine whether the evaluation metric is lower-better or higher-better. You should respond in the following format:

<is_lower_better>
True if the metric is lower-better, False otherwise. 
</is_lower_better>

<explanation>
Your rationale for your decision in 3-4 sentences.
</explanation>
"""

        response = query_llm(self.llm_cfg, messages=[{
            "role": "system", 
            "content": prompt
        }], required_fields=["is_lower_better"], logger=self.llm_logger)

        return "true" in response["is_lower_better"][0].lower()
    
    def _fetch_external_data(self):
        kernels = download_kernels(self.cfg, self.is_lower_better)
        self.logger.info(f"Listed {len(kernels)} kernels, downloading...")
        for kernel in kernels:
            self.logger.info(f"Processing kernel {kernel}...")
            kernel_metadata = get_kernel_metadata(kernel)
            for ref_kernel in kernel_metadata["kernel_sources"]:
                if ref_kernel:
                    self.logger.info(f"Downloading referenced kernel {ref_kernel}...")
                    path = download_kernel_output(self.cfg, ref_kernel)
                    self.datasets[ref_kernel] = Dataset(
                        id=ref_kernel,
                        name=ref_kernel,
                        description=f"Referenced kernel {ref_kernel}",
                        base_path=path
                    )
            for ref_model in kernel_metadata["model_sources"]:
                if ref_model:
                    self.logger.info(f"Downloading referenced model {ref_model}...")
                    path = download_model(self.cfg, ref_model)
                    metadata = get_model_metadata(self.cfg, ref_model)
                    self.datasets[ref_model] = Dataset(
                        id=ref_model,
                        name=metadata["title"],
                        description="Description not available" if "description" not in metadata else metadata["description"],
                        base_path=path
                    )
            for ref_dataset in kernel_metadata["dataset_sources"]:
                if ref_dataset:
                    self.logger.info(f"Downloading referenced dataset {ref_dataset}...")
                    path = download_dataset(self.cfg, ref_dataset)
                    metadata = get_dataset_metadata(self.cfg, ref_dataset)
                    self.datasets[ref_dataset] = Dataset(
                        id=ref_dataset,
                        name=metadata["title"],
                        description=metadata["description"],
                        base_path=path
                    )
        self.kernel_notebooks = kernels

    def _read_jupyter_notebook(self, file_path: Path) -> str:
        """ Read the code and markdown cells from a Jupyter notebook. """

        assert file_path.suffix == ".ipynb", "File must be a Jupyter notebook."

        with open(file_path, "r", encoding='utf-8') as f:
            notebook = json.load(f)

        result = ""
        last_cell_type = "markdown"
        for cell in notebook["cells"]:
            if last_cell_type != cell["cell_type"]:
                result += "\n```\n"
            else:
                result += "\n"
            if cell["cell_type"] == "code":
                result += "".join(cell["source"])
            elif cell["cell_type"] == "markdown":
                result += "".join(cell["source"])
            last_cell_type = cell["cell_type"]
        if last_cell_type == "code":
            result += "\n```\n"
        return result

    def _summarize_public_kernel(self, kernel_path: Path) -> Pipeline:
        # An example of metadata: {
        #  "id": "timoboz/my-awesome-kernel",
        #  "id_no": 12345,
        #  "title": "My Awesome Kernel",
        #  "code_file": "my-awesome-kernel.ipynb",
        #  "language": "python",
        #  "kernel_type": "notebook",
        #  "is_private": "false",
        #  "enable_gpu": "false",
        #  "enable_internet": "false",
        #  "dataset_sources": ["timoboz/my-awesome-dataset"],
        #  "competition_sources": [],
        #  "kernel_sources": [],
        #  "model_sources": [],
        #  "score": 0.95 (optional)
        # }

        self.logger.info(f"Summarizing kernel {kernel_path}")

        metadata = get_kernel_metadata(kernel_path)

        if metadata["kernel_type"] == "notebook":
            content = self._read_jupyter_notebook(kernel_path)
        else:
            with open(kernel_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        if len(content) > 100000:
            content = content[:50000] + "..." + content[-50000:]

        prompt = f"""
You are a professional Kaggle competitor tasked with achieving the highest possible score on the competition. 

<task_desc>\n{self.task_desc}\n</task_desc>

Here is a public kernel published on Kaggle's website. Your goal is to summarize the kernel and provide a holistic overview. You should respond in the following format:

<summary>
A fully detailed description of the kernel. You should report the model architecture, training strategies, inference details and evaluation metrics. The kernel may contain intensive feature engineering and hyperparameter tuning. Report them as well. If it generates checkpoints or references other datasets, you should describe how to load them.
</summary>

<suggestions>
Your suggestions for improving this kernel. You should list the weaknesses and possible future improvements. Do not separate them in multiple suggestion sections. This should be a markdown list. 
</suggestions>

<code>
A representative code segment for this pipeline. You **must include dataset reading / submission generation parts**. This is important for agents to understand how to reproduce the results. If task-specific details such as feature engineering are included, the code segment should contain them as well. You should also include essential parts, including model architecture, training strategies, inference details and evaluation metrics. Omit other non-essential parts and mark them with ellipsis. Do not wrap the code in ```python tags. Include codes for loading checkpoints and datasets.
</code>

<submission>
The name of the submission file. It does not necessarily be {self.submission_name}. Leave this field as None if the code does not produce a submission.
</submission>

Here is the kernel content:
<kernel>\n{content}\n</kernel>

Your response must contain summary, suggestions and code sections.
"""
        def validate_fn(response: dict) -> bool:
            if len(response["summary"]) != 1 or len(response["suggestions"]) != 1 or len(response["code"]) != 1:
                return False
            return True

        response = query_llm(self.llm_cfg, messages=[{
            "role": "system", 
            "content": prompt
        }], required_fields=["summary", "suggestions", "code", "submission"], check_fn=validate_fn, logger=self.llm_logger)

        submission_path = kernel_path.parent / response["submission"][0]
        if "none" in response["submission"][0].lower():
            submission_path = None

        datasets = []
        referenced_private_data = False

        ref_resources = metadata["dataset_sources"] + metadata["model_sources"] + metadata["kernel_sources"]
        for ref_resource in ref_resources:
            if ref_resource:
                datasets.append(self.datasets[ref_resource])
            else:
                referenced_private_data = True

        self.datasets[metadata["id"]] = Dataset(
            id=metadata["id"],
            name=metadata["title"],
            description=response["summary"][0],
            base_path=kernel_path.parent
        )

        metric = WorstMetricValue() if "score" not in metadata else MetricValue(metadata["score"], maximize=not self.is_lower_better)
        
        if submission_path and not submission_path.exists():
            submission_path = None
        
        if not submission_path:
            metric = WorstMetricValue()

        return Pipeline(
            id=metadata["id"],
            title=metadata["title"],
            description=response["summary"][0],
            code=response["code"][0],
            full_code=content,
            metric=metric,
            submission=submission_path,
            output_dir=kernel_path.parent,
            datasets=datasets,
            suggestions=response["suggestions"][0],
            referenced_private_data=referenced_private_data
        )
    
    def _summarize_public_kernels(self):
        for kernel_path in self.kernel_notebooks:
            pipeline = self._summarize_public_kernel(kernel_path)
            self.reports.append(pipeline)
    
    def _get_report_from_id(self, id: str) -> Pipeline | None:
        for report in self.reports:
            if report.id == id:
                return report 
        return None
    
    def _update_report(self, id: str, report: Pipeline):
        for i, report in enumerate(self.reports):
            if report.id == id:
                self.reports[i] = report
                return
    
    def _brainstorm(self):
        prompt = f"""
You are an expert machine learning researcher preparing for the Kaggle competition described below.

<task_desc>\n{self.task_desc}\n</task_desc>

Your job now is to:
- Think creatively and construct at least **4 alternative and highly novel solution paths** that are likely to perform well, especially if combined with careful experimentation.
- Each solution path can be a strategy, pipeline, or method that combines multiple techniques. Try to make them as different as possible from the existing `ideas` and `reports` list.
- After describing each full solution path, **break it down into individual atomic ideas**—these should be the smallest units of implementation (e.g., "use LightGBM for baseline," "normalize input features," "apply stratified K-fold CV").
- Ensure these ideas do not substantially duplicate items already in `ideas`.

<ideas>\n{self._get_ideas_str()}\n</ideas>

<reports>\n{self._get_reports_str()}\n</reports>

Be ambitious but realistic—many ideas can later be tested on a small subset of the data. Focus on novelty, diversity, and decomposability.

You should respond in the following format:

<solution>
<description>A detailed description of this approach. You should explain the novelty of this approach and why it is likely to perform well.</description>
<idea>Description of idea 1 decomposed from this approach.</idea>
<idea>idea 2...</idea>
...
</solution>

<solution>
... other solutions ...
</solution>

...

Make sure all your solutions are well-explained. 
"""
        def validate_fn(response: dict) -> bool:
            if len(response["solution"]) < 4:
                return False
            for solution in response["solution"]:
                try:
                    _ = extract_fields(solution, ["description", "idea"])
                except Exception as e:
                    return False
            return True
        
        response = query_llm(self.llm_cfg, messages=[{
            "role": "system", 
            "content": prompt
        }], required_fields=["solution"], check_fn=validate_fn, logger=self.llm_logger)
        
        for solution in response["solution"]:
            results = extract_fields(solution, ["description", "idea"])
            self.ideas.append(results["description"][0])
            self.ideas.extend(results["idea"])
        
    def _rephrase(self):
        prompt = f"""
You are a machine learning expert. After carefully searching the relevant literature, you have come up with a list of ideas to implement. However, this idea list has some issues. Some ideas are overlapping, you should rephrase and decouple them. You should discard ideas that are irrelevant to the final performance, such as error visualization, etc. 

<ideas>\n{self._get_ideas_str()}\n</ideas>

<reports>\n{self._get_reports_str()}\n</reports>

Please rephrase, merge, and reconstruct the ideas. Do not simplify any technical details of these ideas.

Respond in the following format:

<idea>description of the idea...</idea>
<idea>...</idea>
...

Make sure all your ideas are well-explained.
"""
        response = query_llm(self.llm_cfg, messages=[{
            "role": "system", 
            "content": prompt
        }], required_fields=["idea"], logger=self.llm_logger)

        self.ideas = response["idea"]

    def _get_ideas_str(self) -> str:
        return "\n".join(f"- {idea}" for idea in self.ideas)
    
    def _get_reports_str(self) -> str:
        return "\n".join(report.to_str(full_code=False) for report in self.reports)

    def _generate_pipelines(self) -> list[Draft]:
        prompt = f"""
You are an expert machine learning researcher preparing for the Kaggle competition described below.

<task_desc>\n{self.task_desc}\n</task_desc>

We have collected a list of ideas and reports to help you get the best score. Each report contains the following fields:
- title: the title of the report. 
- description: a summary of the report. This typically contains high-level design choices and the evaluation metrics. Use this to help you understand the report.
- referenced_private_data: a boolean value indicating whether the report references private data. If it does, only use it for ensembling or reference. Do not choose it as a codebase. 
- code: a representative segment of the code.
- metric: the evaluation metric of this pipeline, following the format defined in the task description.
- datasets: a list of datasets referenced in the pipeline. These datasets are also visible to you.
- suggestions: a list of suggestions for improving the pipeline, generated by an LLM.

<ideas>\n{self._get_ideas_str()}\n</ideas>

<reports>\n{self._get_reports_str()}\n</reports>

Your task is to generate a list of pipelines that are likely to achieve the highest possible score. You should generate exactly {self.cfg.agent_num_code_agents} pipelines. **Use Pytorch instead of Tensorflow**. Each pipeline should not overlap with others and as diverse as possible. Your proposed pipelines should include **one pipeline that extends the best method generated so far**, if any provided. Ensure that each pipeline can be trained within 2 hours on a single A6000 with 48GB memory. To gain best performance, you should at least include the best report with the highest score generated so far (if provided) and reference it for ensembling before producing the final submission.

These pipelines will be implemented and executed in separate environments. That is, if you generate 4 pipelines, the last pipeline should not be an ensemble of the first 3 pipelines.

Respond in the following format:

<pipeline>
<title>The title of the pipeline. This should only contain letters, numbers, and spaces. Do not include any other characters. The title should contain 30 characters at most.</title>
<description>An extremely detailed description of the pipeline. Include model architecture, training strategies, hyperparameters, evaluation metrics and input/output details. Read the **submission format** requirements in the task description carefully. The submission format requirement is possible to be different from the training dataset. **THIS IS EXTREMELY IMPORTANT**. Mention in the pipeline descriptions and be sure to include the code that handles the input and output. If any datasets are referenced, explain the structure of each dataset and how to read them. If kernels are referenced, you must describe how to load their checkpoints, whether the submissions files exist, and their evaluation metrics if available. Mention how to compute the evaluation metric.</description>
<datasets>a list of dataset ids referenced in this pipeline, separated by comma. e.g. alice/dataset1,bob/dataset2,etc. You can also include kernel ids for ensembling or loading checkpoints. Do not contain any spaces. If no datasets are referenced, leave this field empty. Do not include dataset {self.cfg.competition_id}</datasets>
<codebase>The codebase of the pipeline. The later implementation agent will use this codebase as a starting point. This field should either be a string exactly matching the id (not the title!) of a report in the reports section. Do NOT choose any reports that reference private data as the codebase.</codebase>
<code>
Python code segment for this pipeline. Make sure to include any important parts, especially those that handle input and output.
</code>
</pipeline>

<pipeline>
... other pipelines ...
</pipeline>

...

Make sure all your pipelines are well-explained. If similar parts are used in multiple pipelines, you should describe them in each pipeline instead of omitting them. 

"""
        def validate_fn(response: dict) -> bool:
            if len(response["pipeline"]) != self.cfg.agent_num_code_agents:
                return False
            for pipeline in response["pipeline"]:
                try: 
                    results = extract_fields(pipeline, ["title", "description", "datasets", "codebase", "code"])
                    datasets = results["datasets"][0].split(",") if results["datasets"][0] else []
                    for dataset in datasets:
                        if dataset not in self.datasets:
                            print(f"Dataset {dataset} not found in datasets.")
                            return False
                    
                    if "none" not in results["codebase"][0].lower():
                        codebase = self._get_report_from_id(results["codebase"][0])
                        if codebase is None:
                            print(f"Codebase {results['codebase'][0]} not found in reports.")
                            return False
                        if codebase.referenced_private_data:
                            print(f"Codebase {results['codebase'][0]} references private data.")
                            return False
                except Exception as e:
                    print(f"Failed to extract fields from pipeline: {e}.")
                    return False
            return True

        response = query_llm(self.llm_cfg, messages=[{
            "role": "system", 
            "content": prompt
        }], required_fields=["pipeline"], check_fn=validate_fn, logger=self.llm_logger)

        drafts = []

        for pipeline in response["pipeline"]:
            results = extract_fields(pipeline, ["title", "description", "datasets", "codebase", "code"])
            datasets = []
            for dataset in results["datasets"][0].split(","):
                if dataset:
                    datasets.append(self.datasets[dataset])

            codebase = None if "none" in results["codebase"][0].lower() else results["codebase"][0]

            codebase_content = None
            if codebase is not None:
                report = self._get_report_from_id(codebase)
                codebase_content = report.full_code

            drafts.append(Draft(
                id=results["title"][0].lower().replace(" ", "-"),
                title=results["title"][0],
                description=results["description"][0],
                datasets=datasets,
                codebase=codebase,
                codebase_content=codebase_content,
                code=results["code"][0],
            ))

        return drafts
    
    def _get_env_name(self, id: str) -> str:
        import hashlib
        return hashlib.md5(id.encode()).hexdigest()[:8]
    
    def _setup_coder_workspace(self, draft: Draft, base_dir: Path):
        """ 
        file structure:
        - input/
            - competition_id/ # the official competition dataset
            - alice/dataset1/ # other public datasets
            - alice/kernel1/  # referenced kernels
        - working/
            - files copied from the codebase
        """
        self.logger.info(f"Setting up coder workspace for {draft.id}...")
        self.logger.info(f"Base dir: {base_dir}")
        (base_dir / "input" / self.cfg.competition_id).mkdir(parents=True, exist_ok=False)
        (base_dir / "working").mkdir(parents=True, exist_ok=False)

        copytree(self.cfg.competition_input_dir, base_dir / "input" / self.cfg.competition_id, use_symlinks=True)
        extract_archives(base_dir / "input" / self.cfg.competition_id)

        for dataset in draft.datasets:
            (base_dir / "input" / dataset.id).mkdir(parents=True, exist_ok=False)
            copytree(dataset.base_path, base_dir / "input" / dataset.id, use_symlinks=True)
        
        source_env_name = self.cfg.agent_base_conda_env_name
        if draft.codebase:
            codebase_report = self._get_report_from_id(draft.codebase)
            source_env_name = self._get_env_name(codebase_report.id)
            assert codebase_report is not None, f"Codebase {draft.codebase} not found in reports."
            copytree(codebase_report.output_dir, base_dir / "working", use_symlinks=False)
        
        # Use shorter name to avoid path length issues
        target_env_name = self._get_env_name(draft.id)
        
        # Clone the environment
        self._clone_conda_environment(source_env_name, target_env_name)
    
    def _setup_reproduce_workspace(self, report: Pipeline, base_dir: Path):
        self.logger.info(f"Setting up reproduce workspace for {report.id}...")
        self.logger.info(f"Base dir: {base_dir}")
        (base_dir / "input" / self.cfg.competition_id).mkdir(parents=True, exist_ok=False)
        (base_dir / "working").mkdir(parents=True, exist_ok=False)

        copytree(self.cfg.competition_input_dir, base_dir / "input" / self.cfg.competition_id, use_symlinks=True)
        extract_archives(base_dir / "input" / self.cfg.competition_id)

        for dataset in report.datasets:
            (base_dir / "input" / dataset.id).mkdir(parents=True, exist_ok=False)
            copytree(dataset.base_path, base_dir / "input" / dataset.id, use_symlinks=True)
        
        copytree(report.output_dir, base_dir / "working", use_symlinks=False)
        
        # For reproduce workspace, always use the base conda environment as source
        # since we're reproducing from scratch
        source_env_name = self.cfg.agent_base_conda_env_name
        target_env_name = self._get_env_name(report.id)
        
        # Check if target environment already exists, if so, we can skip cloning
        target_env_path = self.conda_envs_dir / target_env_name
        if target_env_path.exists():
            self.logger.info(f"Target environment {target_env_path} already exists, skipping clone")
        else:
            # Clone the environment
            self._clone_conda_environment(source_env_name, target_env_name)

    def _start_coder(self, draft: Draft) -> Pipeline:
        coder_cfg = deepcopy(self.cfg)
        self._setup_coder_workspace(draft, coder_cfg.agent_workspace_dir / draft.id)
        coder_cfg.agent_workspace_dir = self.cfg.agent_workspace_dir / draft.id / "working"
        coder = CodeAgent(
            cfg=coder_cfg,
            draft=draft,
            is_lower_better=self.is_lower_better,
            metric_updater=self.metric_updater,
            evaluator=self.evaluator,
            env_name=self._get_env_name(draft.id)
        )
        
        coder_data = {
            "name": draft.title,
            "messages": [("agent", f"Starting work on: {draft.title}")],
            "code": draft.code,
            "output_lines": ["Initializing..."],
        }
        self.code_agents.append(coder_data)
        
        return coder.run()
    
    def _start_coder_with_result(self, draft: Draft, results_list, index: int):
        try:
            result = self._start_coder(draft)
            results_list[index] = result
        except Exception as e:
            import traceback
            self.logger.error(f"Error in coder {draft.id}: {e}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            results_list[index] = None

    def _start_coders(self, pipelines: list[Draft]) -> list[Pipeline]:
        results = [None] * len(pipelines)
        
        threads = []
        for i, pipeline in enumerate(pipelines):
            thread = threading.Thread(
                target=self._start_coder_with_result, 
                args=(pipeline, results, i)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        self.metric_updater.stop()
        return results
    
    def _reproduce_report(self, report: Pipeline) -> Pipeline:
        prepare_cfg = deepcopy(self.cfg)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.cfg.agent_workspace_dir / report.id / f"prepare_{timestamp}"
        self._setup_reproduce_workspace(report, base_dir)
        prepare_cfg.agent_workspace_dir = base_dir / "working"
        prepare = PrepareAgent(
            cfg=prepare_cfg,
            pipeline=report,
            is_lower_better=self.is_lower_better,
            metric_updater=self.metric_updater,
            evaluator=self.evaluator,
            env_name=self._get_env_name(report.id)
        )
        return prepare.run()

    def _reproduce_report_with_result(self, codebase: str, results_list, index: int):
        try:
            report = self._get_report_from_id(codebase)
            pipeline = self._reproduce_report(report)
            results_list[index] = (codebase, pipeline)
        except Exception as e:
            import traceback
            self.logger.error(f"Error in reproducing report {codebase}: {e}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            results_list[index] = (codebase, None)

    def _reproduce_reports(self, pipelines: list[Draft]):
        codebases = []
        for pipeline in pipelines:
            if pipeline.codebase and pipeline.codebase not in codebases:
                if pipeline.codebase not in self.reproduced_reports:
                    codebases.append(pipeline.codebase)
        
        if not codebases:
            return
            
        results = [None] * len(codebases)
        
        threads = []
        for i, codebase in enumerate(codebases):
            thread = threading.Thread(
                target=self._reproduce_report_with_result, 
                args=(codebase, results, i)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # Process results
        for codebase, pipeline in results:
            if pipeline:
                self._update_report(codebase, pipeline)
                self.reproduced_reports.append(codebase)
    
    def _refetch_codebases(self, pipelines: list[Draft]) -> list[Draft]:
        new_pipelines = []
        for pipeline in pipelines:
            new_pipeline = deepcopy(pipeline)
            if pipeline.codebase:
                new_pipeline.codebase_content = self._get_report_from_id(pipeline.codebase).full_code
            new_pipelines.append(new_pipeline)
        return new_pipelines

    def run(self, start_monitor=False):
        # Save initial state and provide monitor instructions
        if start_monitor:
            state_file = self.save_agent_state()
            self.logger.info("🎛️  Agent state saved for monitoring!")
            self.logger.info(f"📊  To monitor progress, run in another terminal:")
            self.logger.info(f"    python monitor_loader.py {state_file}")
        
        if self.evaluator:
            with Status("Setting up evaluator..."):
                self.evaluator.setup()

        for iteration in range(self.cfg.agent_num_iterations):
            self.logger.info("-" * 10 + f" Iteration {iteration} " + "-" * 10)
            with Status("Brainstorming..."):
                self._brainstorm()
            if start_monitor:
                self.save_agent_state()

            with Status("Rephrasing..."):
                self._rephrase()
            if start_monitor:
                self.save_agent_state()

            with Status("Generating pipelines..."):
                pipelines = self._generate_pipelines()

            with Status("Reproducing codebases..."):
                self._reproduce_reports(pipelines)
            pipelines = self._refetch_codebases(pipelines)

            results: list[Pipeline] = []
            def run_coders_and_store_results():
                nonlocal results
                results = self._start_coders(pipelines)
            
            thread = threading.Thread(target=run_coders_and_store_results)
            thread.start()
            self.metric_updater.run()

            thread.join()
            
            for report in results:
                if report is not None:
                    self.reports.append(report)
                    self.reproduced_reports.append(report.id)
                    self.datasets[report.id] = Dataset(
                        id=report.id,
                        name=report.title,
                        description=report.description,
                        base_path=report.output_dir
                    )
            
            if start_monitor:
                self.save_agent_state()
