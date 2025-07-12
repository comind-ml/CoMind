from utils.config import Config
from utils.response import wrap_docs, wrap_ideas
from utils.metric import MetricValue, WorstMetricValue
from backend import FunctionSpec, query
from coder import CodeAgent
from retriever import Retriever

import torch
import logging
import os
import time
from ast import literal_eval
import shutil

from typing import List, cast
from torch.multiprocessing import Queue, Manager
from rich.console import Console 
from rich.table import Table
from rich.live import Live

logger = logging.getLogger("graphml")

class Agent:
    def __init__(self, task_desc: str, cfg: Config):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.ideas = []
        self.reports = []
        self.task_desc = task_desc
        self.start_time = time.time()
        self.public_pipelines = []


        assert torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"Num of detected gpus: {self.gpu_count}")

        self.kernels_retriever     = Retriever(cfg, cfg.doc_base_dir / "kernels")
        self.discussions_retriever = Retriever(cfg, cfg.doc_base_dir / "discussions")

        logger.info("Agent initialized.")
    
    def get_ideas_str(self):
        assert len(self.ideas) > 0, "Ideas list cannot be empty."
        ideas_str = ""
        for i, idea in enumerate(self.ideas):
            ideas_str += f"({i}) {idea}\n"

        return ideas_str

    def get_reports_str(self, expose_code: bool = True):
        if len(self.reports) == 0:
            return "No reports available."
        reports_str = ""
        for i, report in enumerate(self.reports):
            reports_str += "---------- PIPELINE SEPARATOR ----------\n"
            if "pipeline" not in report:
                logger.warning(f"Pipeline not found in report {i}. Skipping...")
                continue

            reports_str += "Pipeline: \n" + report["pipeline"] + "\n"
            if "summary" in report:
                reports_str += "Summary: \n" + report["summary"] + "\n"
            if "suggestions" in report:
                reports_str += "Weaknesses and Suggestions: \n" + report["suggestions"] + "\n"
            reports_str += "Best Metric: " + str(report["metric"]) + "\n"
            
            if expose_code and report["code"] is not None:
                reports_str += "Full code: \n" + report["code"] + "\n"

        return reports_str
    
    def get_public_pipelines_str(self):
        if len(self.public_pipelines) == 0:
            return "No public pipelines available."
        pipelines_str = ""
        for i, pipeline in enumerate(self.public_pipelines):
            pipelines_str += "---------- PIPELINE SEPARATOR ----------\n"
            pipelines_str += f"Public pipeline ({i}): {pipeline}\n"

        return pipelines_str
        

    def get_initial_ideas(self, k: int = 10):
        best_kernels = self.kernels_retriever.get_best_docs(k)
        hotest_kernels = self.kernels_retriever.get_hotest_docs(k)
        self.summarize_kernels(list(set(best_kernels + hotest_kernels)))

        hotest_discussions = self.discussions_retriever.get_hotest_docs(k)
        self.summarize_discussions(hotest_discussions)

    def summarize_discussions(self, docs: List[str]):
        if len(docs) == 0:
            logger.warning("No discussions found for summarization. Skipping...") 
            return
        
        function_spec = FunctionSpec(
            name="extract_ideas",
            json_schema={
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "string",
                        "description": (
                            "required format: python list of strings, each element is a description of an idea extracted from the discussions. e.g. ['idea 1', 'idea 2']. "
                        ),
                    },
                },
                "required": [
                    "ideas",
                ],
            },
            description="Extract ideas from the documents",
        )

        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Your Task": (
                "These are top-ranked public discussions during the competition. Your job now is to:\n"
                "1. Carefully read the following discussions.\n"
                "2. For each discussion, you should decompose it into critical, novel and inspiring ideas that have potential to win this competition. \n"
            ),
            "Public discussions": wrap_docs(docs),
        }

        while True:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    func_spec=function_spec,
                )
            )

            try:
                ideas = literal_eval(response["ideas"])
                assert isinstance(ideas, list), "Pipelines should be a list."
                assert all(isinstance(idea, str) for idea in ideas), "Each pipeline should be a string."

                self.ideas = self.ideas + ideas
            except (SyntaxError, ValueError, AssertionError) as e:
                logger.warning(f"Error parsing pipelines list: {e}")
                continue

            break
        
        logger.info(f"Public ideas: {ideas}")
        return response

    def summarize_kernels(self, docs: List[str]):
        if len(docs) == 0:
            logger.warning("No kernels found for summarization. Skipping...") 
            return

        function_spec = FunctionSpec(
            name="extract_pipeline",
            json_schema={
                "type": "object",
                "properties": {
                    "pipelines": {
                        "type": "string",
                        "description": (
                            "Description of each pipeline, separated by ===SEPARATOR=== mark. \n"
                            "For each pipeline, follow this format:\n"
                            "- Pipeline: A full detailed description of the pipeline, all input/output format, hyperparameters, training settings, model architectures, feature engineering, validation metric, and any other relevant information should be included. **Do not omit any feature engineering details**."
                            "- Code abstract: A representative code segments that captures the essence (including input/output) and novelty of the pipeline. You **MUST** go through all the publicly available code and **include the parts that generate the submission file**. Contain task-specific engineering details. Mark the remainder as ellipses.\n"
                        ),
                    },
                },
                "required": [
                    "pipelines",
                ],
            },
            description="Extract pipelines from the documents",
        )

        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Your Task": (
                "These are top-ranked public scripts during the competition. Your job now is to:\n"
                "1. Carefully read the following scripts.\n"
                "2. For each script, if it's self-contained, i.e., including model arhitecture (if there's a model), training strategies, evaluation, etc., then summarize its pipeline.\n"
                "3. If the pipeline contains technical details, such as extensive feature engineering, hyperparameter tuning, etc., then list them in full detail.\n"
                "4. Select a representative code segment for each pipeline. You must include dataset reading / submission generation parts. If task-specific details such as feature engineering are included, the code segment should contain them as well.\n"
            ),
            "Public scripts": wrap_docs(docs),
        }

        while True:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    func_spec=function_spec,
                )
            )

            try:
                public_pipelines = response["pipelines"].split("===SEPARATOR===")
                public_pipelines = [pipeline.strip() for pipeline in public_pipelines if pipeline.strip()]

            except (SyntaxError, ValueError, AssertionError) as e:
                logger.warning(f"Error parsing pipelines list: {e}")
                continue

            break
        
        self.public_pipelines = public_pipelines
        logger.info(self.get_public_pipelines_str())

        return response
    
    def reconstruct_ideas(self, max_tries: int = 3):
        assert len(self.ideas) > 0, "Ideas list cannot be empty."
        
        function_spec = FunctionSpec(
            name="reconstruct_ideas",
            json_schema={
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "string",
                        "description": "required format: python list of strings, each element is an idea. e.g. ['idea 1', 'idea 2']",
                    },
                },
                "required": [
                    "ideas",
                ],
            },
            description="Decompose, merge, and reconstruct the ideas",
        )

        prompt = (
            "You are a machine learning expert. After carefully searching the relevant literature, you have come up with a list of ideas to implement. However, this idea list has some issues: \n"
            "- Some ideas are too similar and should be merged into one. \n"
            "- Some ideas are overlapping, you should rephrase and decouple them. \n"
            "- You should discard ideas that are irrelevant to the final performance, such as error visualization, etc. \n"
            "You should refer to the Reports and Public pipelines section for the latest updates on the ideas and previous pipelines. \n"
            "Do not decompose highly self-contained and promising ideas. \n"
            "Please decompose, merge, and reconstruct the ideas. \n"
            f"Ideas: {self.get_ideas_str()}\n"
            f"Reports: {self.get_reports_str(expose_code=False)}\n"
            f"Public pipelines: {self.get_public_pipelines_str()}\n"
        )

        for _ in range(max_tries):
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=function_spec,
                    model=self.acfg.brainstorm.model,
                    temperature=self.acfg.brainstorm.temp,
                )
            )

            if response["ideas"] is not None:
                try:
                    ideas_list = literal_eval(response["ideas"])
                    self.ideas = []
                    for idea in ideas_list:
                        self.ideas.append(idea)
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Error parsing ideas list: {e}")
                    continue

                break

            else:
                logger.warning("No ideas list found in the response. Retrying...")

    def get_pipelines(self, num_pipes: int = 4) -> List[str]:
        logger.info(f"Idea list: {self.get_ideas_str()}")

        function_spec = FunctionSpec(
            name="submit_pipelines",
            json_schema={
                "type": "object",
                "properties": {
                    "submit_pipelines": {
                        "type": "string",
                        "description": "Descriptions and codes of pipelines, separated each pipeline by ===SEPARATOR=== mark. For each pipeline, Attach code that captures its essential. **You must include the code in public pipelines that handles input and output, and if there are parts of the public pipelines that are similar to the current pipeline, you should include them as well.**",
                    },
                },
                "required": [
                    "submit_pipelines",
                ],
            },
            description="Propose a list of pipelines based on the ideas and reports",
        )

        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Ideas": self.get_ideas_str(),
            "Reports": self.get_reports_str(),
            "Public pipelines": self.get_public_pipelines_str(),
            "Your Task": (
                "1. Carefully read the reports provided above. \n"
                f"2. Based on the ideas and reports, propose **{num_pipes} promising self-contained pipelines** that are likely to perform well. \n"
                "3. The Public pipelines section contains top-ranked public pipelines during the competition. Use them as reference to polish your pipelines. \n"
                "4. Each pipeline should not overlap with others. Your proposed pipelines should include **one baseline pipeline that uses well-known methods but is robust and relatively easy to implement**. You should reinforce public pipelines and previous pipelines based on their reports (if provided). \n"
                "5. Ensure that each pipeline can be trained within 2 hours on a single A6000 with 48GB memory. \n"
                "6. Read the **submission format** requirements in the task desciption carefully. The format requirement is possible to be different from the training dataset. **THIS IS EXTREMELY IMPORTANT**. Mention in the pipeline descriptions and be sure to include the code that handles the input and output. \n"
                "7. DO NOT USE LIGHTGBM, XGBOOST, CATBOOST, or any other tree-based models in the pipelines. \n"
                "8. DO NOT USE tensorflow, use pytorch instead. \n"
            )
        }

        while True:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=function_spec,
                    model=self.acfg.brainstorm.model,
                    temperature=self.acfg.brainstorm.temp,
                )
            )

            logger.info(f"Pipeline response:\n{response}")

            try:
                pipelines = response["submit_pipelines"].split("===SEPARATOR===")
                pipelines = [pipeline.strip() for pipeline in pipelines if pipeline.strip()]
                assert len(pipelines) == num_pipes, f"Expected {num_pipes} pipelines, but got {len(pipelines)}."

            except (SyntaxError, ValueError, AssertionError) as e:
                logger.warning(f"Error parsing pipelines list: {e}")
                continue

            break
    
        return pipelines

    def brainstorm(self):
        prompt = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.task_desc,
            "Your Task": (
                "I already have a list of ideas that partially explore how to approach this competition. Your job now is to:\n"
                "1. Think creatively and construct at least **4 alternative and highly novel solution paths** that are likely to perform well, especially if combined with careful experimentation. \n"
                "2. Each solution path can be a strategy, pipeline, or method that combines multiple techniques. Try to make them as different as possible from the existing `ideas` list. \n"
                "3. After describing each full solution path, **break it down into individual atomic ideas**—these should be the smallest units of implementation (e.g., “use LightGBM for baseline,” “normalize input features,” “apply stratified K-fold CV”).\n"
                "4. Ensure these ideas do not substantially duplicate items already in `ideas`.\n"
                "5. Refer to the `Reports` section for the latest updates on the ideas and previous pipelines.\n"
            ),
            "Ideas": self.get_ideas_str(),
            "Reports": self.get_reports_str(expose_code=False),
            "Public pipelines": self.get_public_pipelines_str(),
            "Instructions": {
                "Response Format": (
                    "Format your output like this (one line, one idea):\n"
                    "<Your understanding of the task and explanation of your approaches>\n"
                    "===SOLUTION_PATH_1===\n"
                    "<Description of this approach>\n"
                    "- atomic idea 1\n"
                    "- atomic idea 2\n"
                    "- atomic idea 3\n"
                    "...\n"
                    "===SOLUTION_PATH_2===\n"
                    "...\n"
                    "===SOLUTION_PATH_3===\n"
                    "...\n"
                ),
                "Reminder": "Be ambitious but realistic—many ideas can later be tested on a small subset of the data. Focus on novelty, diversity, and decomposability. Ready? Start."
            }
        }

        response = query(
            system_message=prompt,
            user_message=None,
            model=self.acfg.brainstorm.model,
            temperature=self.acfg.brainstorm.temp,
        )

        logger.info(f"Brainstorming response:\n{response}")

        for line in response.split("\n"):
            if line.startswith("-"):
                self.ideas.append(line[2:].strip())

    def run(self):
        self.get_initial_ideas()
        self.metric = WorstMetricValue()

        for _ in range(self.acfg.iterations):
            self.brainstorm()
            logger.info("Brainstorming completed.")
            max_threads = self.acfg.max_threads

            self.reconstruct_ideas()
            pipelines = self.get_pipelines(num_pipes=max_threads)
            for pipeline in pipelines:
                logger.info(f"Pipeline: {pipeline}")

            def render_table(status_dict):
                table = Table(title="Code Agents Status Panel", box=None, min_width=40)
                table.add_column("Agent ID", style="cyan", no_wrap=True)
                table.add_column("Status", style="magenta")

                for name, status in sorted(status_dict.items()):
                    table.add_row(str(name), status)

                return table

            with Manager() as manager:
                processes = []
                result_queue = Queue()
                status_dict = manager.dict()
                best_metric = manager.dict()

                write_lock = manager.Lock()

                best_metric["value"] = self.metric.value
                best_metric["maximize"] = self.metric.maximize

                for i, pipeline in enumerate(pipelines):
                    logger.info(f"Running group {i + 1}/{max_threads} on GPU {i}...")
                    process = torch.multiprocessing.Process(
                        target=self.run_coder, 
                        args=(i, pipeline, result_queue, best_metric, write_lock, status_dict)
                    )
                    process.start()
                    processes.append(process)
                
                console = Console()
                with Live(render_table(status_dict), console=console, refresh_per_second=2) as live:
                    while any(p.is_alive() for p in processes):

                        finished_processes = ["Finished" in status for status in status_dict.values()]
                        if all(finished_processes):
                            break

                        live.update(render_table(status_dict))
                        time.sleep(0.5)

                    # Final update
                    live.update(render_table(status_dict))
                    logger.info("\n✅ All code agents have completed!")
                
                for p in processes:
                    p.join()
                
                self.metric = MetricValue(
                    value=best_metric["value"],
                    maximize=best_metric["maximize"],
                )

                while not result_queue.empty():
                    result = result_queue.get(timeout=1)
                    if result is not None:
                        self.reports.append(result["report"])
        
    def run_coder(self, agent_id, pipeline: str, result_queue: Queue, best_metric, write_lock, status_dict):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_id % self.gpu_count)
        logger.info(f"Running on GPU {agent_id % self.gpu_count}...")
        status_dict[agent_id] = "Initializing CodeAgent..."

        coder = CodeAgent(
            agent_id=agent_id, 
            task_desc=self.task_desc, 
            cfg=self.cfg, 
            global_start_time=self.start_time,
            global_metric=best_metric,
            global_lock=write_lock,
            status_dict=status_dict
        )
        result = coder.run(pipeline)

        if result is not None:
            result_queue.put(result)




    

    