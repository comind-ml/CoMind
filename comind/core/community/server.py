import threading
import uuid
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import time
from copy import deepcopy

from flask import Flask, request, jsonify
from pydantic import SecretStr

from comind.core.agent.metric import MetricValue, WorstMetricValue
from comind.core.community.dashboard import Dashboard
from comind.core.config.comm_config import CommConfig
from comind.core.config.llm_config import LLMConfig
from comind.core.config.task_config import TaskConfig
from comind.llm.llm import LLM
from comind.core.assets.fn_specs import propose_idea_func_spec, submit_pipeline_func_spec, metric_direction_func_spec
from comind.utils.generic import get_timestamp, read_file_content, read_kernels_from_path, read_discussions_from_path
from comind.utils.data_preview import generate as generate_data_preview
from comind.core.logger import logger

TOP_K = 5

class CommunityServer:
    def __init__(self, comm_config: CommConfig, llm_config: LLMConfig):
        self.comm_config = deepcopy(comm_config)
        self.llm_config = deepcopy(llm_config)
        
        self.proposer_llm = LLM(self.llm_config)
        self.feedback_llm = LLM(self.llm_config, keep_history=False)  
        self.dashboard = Dashboard()
        
        self.problem_desc = read_file_content(Path(self.comm_config.problem_desc_path))
        self.is_lower_better = self._determine_metric_direction()

        logger.info(f"Problem description: {self.problem_desc}")
        
        prompt_path = Path(__file__).parent.parent / "assets" / "community_prompt.json"
        with open(prompt_path) as f:
            self.prompts = json.load(f)

        self.ideas: List[str] = []
        self.reports: List[Dict[str, Any]] = []
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.best_metric = WorstMetricValue(maximize=not self.is_lower_better)

        self.lock = threading.Lock()
        self._summarize_initial_artifacts()
        self._stop_event = threading.Event()
        self._start_timestamp = get_timestamp()

    def _cleanup_stale_agents(self):
        """Periodically checks for and removes stale agents."""
        while not self._stop_event.is_set():
            self._stop_event.wait(60)  # Check every 60 seconds
            if self._stop_event.is_set():
                break

            timeout_seconds = 5 * 60  # 5 minutes
            stale_agents = self.dashboard.get_stale_agents(timeout_seconds)

            if stale_agents:
                with self.lock:
                    for agent_id in stale_agents:
                        if agent_id in self.agents:
                            logger.info(f"Agent {agent_id} timed out. Removing its data.")
                            del self.agents[agent_id]
                        
                        # Also remove from dashboard
                        self.dashboard.remove_agent(agent_id)
                        logger.info(f"Removed stale agent {agent_id} from dashboard.")

    def _determine_metric_direction(self) -> bool:
        """Ask the LLM to determine if the evaluation metric should be minimized (lower is better)."""
        prompt = (
            f"Based on the following competition description, determine whether the evaluation metric should be minimized (lower values are better) or maximized (higher values are better).\n\n"
            f"Competition Description:\n{self.problem_desc}\n\n"
            f"Consider the type of task (classification, regression, etc.) and the specific evaluation metric mentioned. "
            f"Common metrics that should be minimized include: MSE, RMSE, MAE, cross-entropy loss, log loss. "
            f"Common metrics that should be maximized include: accuracy, F1-score, AUC, precision, recall, IoU, mAP."
        )
        
        try:
            response = self.feedback_llm.chat(prompt, function_spec=metric_direction_func_spec)
            is_lower_better = response.get("is_lower_better", False)
            reasoning = response.get("reasoning", "No reasoning provided")
            logger.info(f"Metric direction determined: lower_is_better={is_lower_better}. Reasoning: {reasoning}")
            return is_lower_better
        except Exception as e:
            logger.warning(f"Failed to determine metric direction via LLM: {e}. Defaulting to lower_is_better=False")
            return False

    def _format_prompt(self, prompt_dict: Dict[str, str]) -> str:
        """Formats a dictionary prompt into a string for the LLM."""
        formatted_str = []
        for key, value in prompt_dict.items():
            formatted_str.append(f"## {key}\n{value}")
        return "\n\n".join(formatted_str)

    def _summarize_initial_artifacts(self):
        logger.info("Summarizing initial artifacts from discussions and kernels...")
        
        # Summarize Discussions in a batch
        if self.comm_config.discussions_path:
            discussions = read_discussions_from_path(Path(self.comm_config.discussions_path))
            if not discussions:
                logger.warning("No discussions found to summarize.")
            else:
                discussions.sort(key=lambda x: x.get('votes', 0), reverse=True)
                top_discussions = discussions[:TOP_K]
                logger.info(f"Found {len(discussions)} discussions, processing top {len(top_discussions)}")
                
                def wrap_docs(docs: List[Dict[str, Any]]) -> str:
                    formatted_docs = []
                    for d in docs:
                        formatted_docs.append(f"---\nTitle: {d.get('title')}\nVotes: {d.get('votes')}\n\n{d.get('content')}\n---")
                    return "\n\n".join(formatted_docs)

                prompt_dict = {
                    "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
                    "Task Description": self.problem_desc,
                    "Your Task": (
                        "These are top-ranked public discussions during the competition. Your job now is to:\n"
                        "1. Carefully read the following discussions.\n"
                        "2. For each discussion, you should decompose it into critical, novel and inspiring ideas that have potential to win this competition. \n"
                    ),
                    "Public discussions": wrap_docs(top_discussions),
                }
                
                prompt = self._format_prompt(prompt_dict)
                logger.info(f"Sending discussions summarization request to LLM (prompt length: {len(prompt)} chars)")
                try:
                    idea_response = self.feedback_llm.chat(prompt, function_spec=propose_idea_func_spec)
                    ideas_found = idea_response.get("ideas", [])
                    self.ideas.extend(ideas_found)
                    logger.info(f"Successfully extracted {len(ideas_found)} ideas from discussions")
                except Exception as e:
                    logger.warning(f"Failed to summarize discussions batch: {e}")

        # Summarize Kernels in a batch
        if self.comm_config.kernels_path:
            kernels = read_kernels_from_path(Path(self.comm_config.kernels_path))
            if not kernels:
                logger.warning("No kernels found to summarize.")
            else:
                kernels_by_vote = sorted(kernels, key=lambda x: x.get('votes', 0), reverse=True)
                kernels_with_score = [k for k in kernels if isinstance(k.get('bestPublicScore'), (int, float))]
                kernels_by_score = sorted(kernels_with_score, key=lambda x: x['bestPublicScore'], reverse=True)

                top_kernels_dict = {}
                for k in kernels_by_vote[:TOP_K]: top_kernels_dict[k['id']] = k
                for k in kernels_by_score[:TOP_K]: top_kernels_dict[k['id']] = k
                
                top_kernels_list = list(top_kernels_dict.values())
                logger.info(f"Found {len(kernels)} kernels, processing top {len(top_kernels_list)} unique kernels")

                formatted_kernels = []
                for k in top_kernels_list:
                    formatted_kernels.append(
                        f"--- Script Start ---\n"
                        f"Title: {k.get('title')}\n"
                        f"Content:\n{k.get('content')}\n"
                        f"--- Script End ---"
                    )
                kernel_context = "\n\n".join(formatted_kernels)
                
                your_task_instruction = (
                    "These are top-ranked public scripts during the competition. Your job now is to:\n"
                    "1. Carefully read the following scripts.\n"
                    "2. For each script, if it's self-contained, i.e., including model arhitecture (if there's a model), training strategies, evaluation, etc., then summarize its pipeline.\n"
                    "3. If the pipeline contains technical details, such as extensive feature engineering, hyperparameter tuning, etc., then list them in full detail.\n"
                    "4. Select a representative code segment for each pipeline. You must include dataset reading / submission generation parts. If task-specific details such as feature engineering are included, the code segment should contain them as well.\n"
                    "5. Provide a clear description of each pipeline's approach and methodology."
                )
                prompt = self.prompts["summarize_kernels_batch"].format(
                    problem_desc=self.problem_desc,
                    your_task=your_task_instruction,
                    scripts=kernel_context
                )
                
                logger.info(f"Sending kernels summarization request to LLM (prompt length: {len(prompt)} chars)")
                try:
                    llm_summary = self.feedback_llm.chat(prompt, function_spec=submit_pipeline_func_spec)
                    summarized_pipelines = llm_summary.get("pipelines", [])
                    
                    # Save all pipeline summaries without title matching
                    for i, pipeline_summary in enumerate(summarized_pipelines):
                        # Add score from the corresponding kernel if available
                        if i < len(top_kernels_list):
                            pipeline_summary['score'] = top_kernels_list[i].get('bestPublicScore')
                        else:
                            pipeline_summary['score'] = None
                        self.reports.append(pipeline_summary)
                    
                    logger.info(f"Successfully extracted {len(summarized_pipelines)} pipeline reports from kernels")
                except Exception as e:
                    logger.warning(f"Failed to summarize kernels batch: {e}")

        logger.info(f"Initialization complete. Found {len(self.reports)} initial reports and {len(self.ideas)} ideas.")

    def _get_ideas_str(self) -> str:
        if not self.ideas: return "N/A"
        return "- " + "\n- ".join(self.ideas)

    def _get_agent_reports_str(self, agent_reports: List[Dict[str, Any]]) -> str:
        if not agent_reports: return "N/A"
        reports_details = []
        for r in agent_reports:
            report_str = (
                f"--- Agent Report (Score: {r.get('score', 'N/A')}) ---\n"
                f"Pipeline: {r.get('pipeline', 'N/A')}\n"
                f"Code abstract: {r.get('code_abs', 'N/A')}\n"
                f"Summary: {r.get('summary', 'N/A')}\n"
                f"Suggestions for Improvement: {r.get('suggestions', 'N/A')}\n"
                f"------------------------------------"
            )
            reports_details.append(report_str)
        return "\n\n".join(reports_details)

    def _get_public_pipelines_str(self, public_pipelines: List[Dict[str, Any]]) -> str:
        if not public_pipelines: return "N/A"
        reports_details = []
        for r in public_pipelines:
            report_str = (
                f"--- Public Kernel (Score: {r.get('score', 'N/A')}) ---\n"
                f"Pipeline: {r.get('pipeline', 'N/A')}\n"
                f"Code abstract: {r.get('code_abs', 'N/A')}\n"
                f"-----------------------------------"
            )
            reports_details.append(report_str)
        return "\n\n".join(reports_details)

    def _generate_new_pipeline(self) -> str:
        with self.lock:
            active_pipelines_desc = [agent_data.get('pipeline', 'N/A') for agent_data in self.agents.values()]
            agent_reports = [r for r in self.reports if 'summary' in r]
            public_pipelines = [r for r in self.reports if 'summary' not in r]

        num_pipes = 1
        prompt_dict = {
            "Introduction": "You are an expert machine learning researcher preparing for the Kaggle competition described below.",
            "Task Description": self.problem_desc,
            "Ideas": self._get_ideas_str(),
            "Reports": self._get_agent_reports_str(agent_reports),
            "Public pipelines": self._get_public_pipelines_str(public_pipelines),
            "Pipelines currently in progress (DO NOT REPEAT)": "- " + "\n- ".join(active_pipelines_desc),
            "Your Task": (
                f"1. Carefully read the reports provided above. \n"
                f"2. Based on the ideas and reports, propose **{num_pipes} promising self-contained pipeline** that is likely to perform well. \n"
                "3. The Public pipelines section contains top-ranked public pipelines during the competition. Use them as reference to polish your pipelines. \n"
                "4. Each pipeline should not overlap with others, especially those already in progress. Your proposed pipelines should include **one baseline pipeline that uses well-known methods but is robust and relatively easy to implement**. You should reinforce public pipelines and previous pipelines based on their reports (if provided). \n"
                "5. Ensure that each pipeline can be trained within 2 hours on a single A6000 with 48GB memory. \n"
                "6. Read the **submission format** requirements in the task desciption carefully. The format requirement is possible to be different from the training dataset. **THIS IS EXTREMELY IMPORTANT**. Mention in the pipeline descriptions and be sure to include the code that handles the input and output. \n"
                "7. DO NOT USE LIGHTGBM, XGBOOST, CATBOOST, or any other tree-based models in the pipelines. \n"
                "8. DO NOT USE tensorflow, use pytorch instead. \n"
                "9. Your response MUST be a function call. The 'code_abs' field should not be empty."
            )
        }
        
        prompt = self._format_prompt(prompt_dict)
        
        logger.info(f"Generating pipeline with prompt length: {len(prompt)} chars")
        logger.debug(f"Pipeline generation prompt: {prompt[:500]}...")  # Log first 500 chars
        
        try:
            response = self.proposer_llm.chat(prompt, function_spec=submit_pipeline_func_spec)
            logger.info(f"LLM response received: {response}")
            
            generated_pipelines = response.get("pipelines", [])
            logger.info(f"Generated {len(generated_pipelines)} pipelines")
            
            if generated_pipelines:
                pipeline_desc = generated_pipelines[0].get("pipeline", "")
                code_abs = generated_pipelines[0].get("code_abs", "")
                
                logger.info(f"Pipeline description length: {len(pipeline_desc)}")
                logger.info(f"Code abstract length: {len(code_abs)}")
                
                if not pipeline_desc:
                    logger.warning("Pipeline description is empty!")
                if not code_abs:
                    logger.warning("Code abstract is empty!")
                
                response = "Generated pipeline description is empty"
                if pipeline_desc:
                    response = f"Pipeline description: {pipeline_desc}\nCode abstract: {code_abs}"
                return response
            else:
                logger.warning("No pipelines generated by LLM")
                # Re-raise the exception to get more information
                raise Exception("LLM returned empty pipelines list")
                
        except Exception as e:
            logger.error(f"Error generating pipeline via LLM: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the exception to get the full error in the server logs
            raise

    def serve(self):
        app = Flask(__name__)

        # Start the stale agent cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_stale_agents, daemon=True)
        cleanup_thread.start()

        @app.route("/config", methods=["POST"])
        def fetch_config():
            try:
                data = request.json
                agent_id = data.get("agent_id", str(uuid.uuid4()))
                
                logger.info(f"Fetching config for agent: {agent_id}")
                self.dashboard.add_agent(agent_id)
                
                logger.info("Generating new pipeline...")
                pipeline_desc = self._generate_new_pipeline()
                logger.info(f"Pipeline generated successfully: {pipeline_desc[:100]}...")
                
                with self.lock:
                    self.agents[agent_id] = {"pipeline": pipeline_desc, "status": "generating_pipeline"}
                
                data_overview_str = "No data overview available."
                if self.comm_config.dataset_path:
                    dataset_path = Path(self.comm_config.dataset_path)
                    if dataset_path.exists() and dataset_path.is_dir():
                        logger.info("Generating data overview...")
                        data_overview_str = generate_data_preview(dataset_path)

                task_config = TaskConfig(
                    agent_id=agent_id,
                    task_desc=self.problem_desc,
                    pipeline=pipeline_desc,
                    data_overview=data_overview_str,
                )
                logger.info(f"Task config created successfully for agent: {agent_id}")
                return jsonify(task_config.dict())
            except Exception as e:
                logger.error(f"Error in fetch_config: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        @app.route("/status", methods=["POST"])
        def upload_status():
            data = request.json
            agent_id = data.get("agent_id")
            status_info = data.get("status")
            if agent_id and status_info:
                self.dashboard.update_agent_status(agent_id, status_info)
                return jsonify({"status": "ok"})
            return jsonify({"error": "Missing agent_id or status"}), 400
        
        @app.route("/heartbeat", methods=["POST"])
        def heartbeat():
            """Endpoint for agents to send heartbeat signals"""
            data = request.json
            agent_id = data.get("agent_id")
            if agent_id:
                # Update last seen time without changing status
                current_status = self.dashboard.agent_statuses.get(agent_id, {"state": "heartbeat", "query": "Sending heartbeat..."})
                self.dashboard.update_agent_status(agent_id, current_status)
                return jsonify({"status": "ok"})
            return jsonify({"error": "Missing agent_id"}), 400

        @app.route("/submission", methods=["POST"])
        def upload_submission():
            agent_id = request.form.get("agent_id")
            metric_value = float(request.form.get("metric"))
            
            if not agent_id or metric_value is None:
                return jsonify({"error": "Missing agent_id or metric"}), 400
            
            with self.lock:
                current_metric = self.best_metric.__class__(metric_value, maximize=not self.is_lower_better)
                
                is_new_best = current_metric > self.best_metric

                if is_new_best:
                    self.best_metric = current_metric
                    submission_file = request.files.get('file')
                    if submission_file:
                        save_path = Path("./submissions") / self._start_timestamp
                        save_path.mkdir(exist_ok=True)
                        submission_file.save(save_path / f"best_submission_{agent_id}.csv")
                    logger.info(f"New best metric from {agent_id}: {self.best_metric.value}")

                self.dashboard.update_metrics(agent_id, current_metric, self.best_metric)

            return jsonify({"status": "ok", "is_best": is_new_best})

        @app.route("/report", methods=["POST"])
        def upload_report():
            data = request.json
            agent_id = data.get("agent_id")
            report = data.get("report")
            if agent_id and report:
                score_dict = report.get("score")
                if isinstance(score_dict, dict):
                    metric_value = score_dict.get("value")
                    maximize = score_dict.get("maximize")
                    
                    if metric_value is None:
                        report["score"] = WorstMetricValue(maximize=maximize)
                    else:
                        report["score"] = MetricValue(value=metric_value, maximize=maximize)

                with self.lock:
                    self.reports.append(report)
                logger.info(f"Received final report from agent {agent_id}.")
                return jsonify({"status": "ok"})
            return jsonify({"error": "Missing agent_id or report"}), 400
        
        self.dashboard.start()
        logger.info(f"Community server starting at http://{self.comm_config.host}:{self.comm_config.port}")
        
        try:
            from waitress import serve
            serve(app, host=self.comm_config.host, port=self.comm_config.port, _quiet=True)
        finally:
            self._stop_event.set()
            self.dashboard.stop()