import asyncio
import json
import time
import requests
import subprocess

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from comind.core.logger import logger
from comind.core.agent.status import AgentState, ExecutionInfo, Status
from comind.utils.prompt import trim_long_string
from comind.core.assets.registry import Registry, embed
from comind.core.agent.metric import MetricValue, WorstMetricValue
from comind.core.config import TaskConfig, DockerConfig, LLMConfig, AgentConfig
from comind.environment.docker import Docker
from comind.llm.llm import LLM
from comind.core.assets.fn_specs import review_func_spec, propose_code_func_spec, report_func_spec

class Agent:
    def __init__(
        self, 
        agent_config: AgentConfig,
        llm_config: LLMConfig,
        docker_config: DockerConfig,
    ):
        self.agent_id = agent_config.agent_id
        self.task_config = None
        self.agent_config = deepcopy(agent_config)
        self.llm_config = deepcopy(llm_config)
        self.docker_config = deepcopy(docker_config)

        # Add missing attributes from agent_config
        self.api_url = agent_config.api_url
        self.working_dir = agent_config.working_dir

        self.current_step = 0 
        self.start_time = time.time()

        self.executor = Docker(self.docker_config)
        self.llm = LLM(self.llm_config)
        self.feedback_llm = LLM(self.llm_config, keep_history=False)
        self.best_metric = WorstMetricValue()
        self.best_code = None
        
        # Heartbeat mechanism
        self._heartbeat_running = False
        self._heartbeat_thread = None

        # Detect GPU info and fill the gpu_info field
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                self.gpu_info = result.stdout.strip()
            else:
                self.gpu_info = "No GPU detected or nvidia-smi not available"
        except Exception:
            self.gpu_info = "No GPU detected or nvidia-smi not available"

        prompt_path = Path(__file__).parent.parent / "assets" / "agent_prompt.json"
        with open(prompt_path) as f:
            self.prompts = json.load(f)

    def _post_request(self, url: str, data: Dict[str, Any]):
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"Failed to post request to {url}: {response.status_code} {response.text}")

        return response.json()
    
    def _fetch_task_config(self):
        assert self.task_config is None 
        request_url = self.api_url + "/config"
        data = {
            "agent_id": str(self.agent_id),
        }
        response = self._post_request(request_url, data)
        self.task_config = TaskConfig(**response)

        if str(self.agent_id) != str(self.task_config.agent_id):
            raise Exception(f"Agent ID mismatch: Expected {self.agent_id}, got {self.task_config.agent_id}")

    def _upload_report(self, report: Dict[str, Any]):
        request_url = self.api_url + "/report"
        data = {
            "agent_id": self.task_config.agent_id,
            "report": report,
        }
        response = self._post_request(request_url, data)
        return response

    def _upload_status(self, status: Status):
        """
        Uploads the current status to the community server.
        """
        try:
            request_url = self.api_url + "/status"
            data = {
                "agent_id": str(self.task_config.agent_id if self.task_config else self.agent_id),
                "status": status.to_dict(),
            }
            response = self._post_request(request_url, data)
            return response
        except Exception as e:
            logger.warning(f"Failed to upload status: {e}")
    
    def _send_heartbeat(self):
        """Send heartbeat to server to indicate agent is still alive"""
        try:
            request_url = self.api_url + "/heartbeat"
            data = {
                "agent_id": str(self.task_config.agent_id if self.task_config else self.agent_id),
            }
            response = self._post_request(request_url, data)
            return response
        except Exception as e:
            logger.debug(f"Failed to send heartbeat: {e}")  # Use debug level to avoid spam
    
    def _start_heartbeat(self):
        """Start the heartbeat thread"""
        import threading
        import time
        
        def heartbeat_loop():
            while self._heartbeat_running and self.task_config:
                time.sleep(30)  # Send heartbeat every 30 seconds
                if self._heartbeat_running:
                    self._send_heartbeat()
        
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat mechanism started")
    
    def _stop_heartbeat(self):
        """Stop the heartbeat thread"""
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1)
        logger.info("Heartbeat mechanism stopped")

    def _upload_submission(self, file_path: Path, metric: MetricValue):
        """
        Uploads a submission file and metric to the community server.
        """
        try:
            request_url = self.api_url + "/submission"
            
            # Prepare form data
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'text/csv')}
                data = {
                    'agent_id': str(self.task_config.agent_id if self.task_config else self.agent_id),
                    'metric': str(metric.value if metric.value is not None else 0.0)
                }
                
                # Use requests for multipart form data
                response = requests.post(request_url, data=data, files=files)
                
                if response.status_code != 200:
                    raise Exception(f"Failed to upload submission: {response.status_code} {response.text}")
                
                result = response.json()
                logger.info(f"Submission uploaded successfully. Is best: {result.get('is_best', False)}")
                return result
                
        except Exception as e:
            logger.warning(f"Failed to upload submission: {e}")

    def _fetch_report(self):
        """
        Generates and uploads a final report when the task is finished.
        """
        logger.info("Generating final report...")
        prompt = "Please summarize the results and submit a comprehensive report."
        status = Status(
            state=AgentState.GENERATING_REPORT,
            query=prompt,
        )
        self._upload_status(status)

        report = self.llm.chat(prompt, function_spec=report_func_spec)
        report["code_abs"] = self.best_code
        report["score"] = self.best_metric.to_dict()
        self._upload_report(report)
        logger.info("Final report submitted.")

    def _query_code(self, exec_result: str | None) -> tuple[str, str]:
        """
        Queries the LLM for the next piece of code to execute.
        """
        if self.current_step == 1:
            template = self.prompts["task_description"]
            info = Registry(
                desc=self.task_config.task_desc,
                pipeline=self.task_config.pipeline,
                data_overview=self.task_config.data_overview,
                time_limit_per_step=self.agent_config.time_limit_per_step,
                total_time_limit=self.agent_config.total_time_limit,
                gpu_info=self.gpu_info,
            )
        else:
            template = self.prompts["next_step_suggestion"]
            info = Registry(
                remain_steps=self.agent_config.total_steps - self.current_step,
                remain_time=self.agent_config.total_time_limit - (time.time() - self.start_time),
                execution_review=exec_result,
            )
        
        prompt = embed(template, info)

        status = Status(
            state=AgentState.QUERYING_LLM,
            query=prompt,
        )
        self._upload_status(status)
        
        response = self.llm.chat(prompt, function_spec=propose_code_func_spec)
        
        code = response.get("code", "")
        explanation = response.get("explanation", "")

        return code, explanation

    def _parse_exec_result(self, code: str, explanation: str, output: str, exec_time: float) -> str:
        info = Registry(
            code=code,
            output=trim_long_string(output) + f"\n\n[System] Execution time: {exec_time:.2f}s",
            explanation=explanation,
        )

        template = self.prompts["execution_review"]
        prompt = embed(template, info)

        status = Status(
            state=AgentState.ANALYZING,
            query=prompt,
            execution_info=ExecutionInfo(
                code=code,
                explanation=explanation,
                elapsed_time=exec_time,
                console_output=output,
            ),
        )

        logger.info(f"Execution status: {status}")

        self._upload_status(status)

        feedback = self.feedback_llm.chat(prompt, function_spec=review_func_spec)

        metric = feedback.get("metric", None)

        if not isinstance(metric, (int, float)) or not (self.working_dir / "submission.csv").exists():
            metric = WorstMetricValue()
        else:
            metric = MetricValue(metric, maximize=not feedback["is_lower_better"])
        
        logger.info(f"Execution ended with metric: {metric}")
        
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_code = code
            self._upload_submission(self.working_dir / "submission.csv", metric)
        
        response = (
            f"Terminal output (truncated): {feedback['output_abs']}\n"
            f"Execution summary: {feedback['summary']}\n"
            f"Execution time: {exec_time:.2f}s\n"
        )

        return response
    
    async def _execute_code(self, code: str, explanation: str) -> str:
        status = Status(
            state=AgentState.EXECUTING,
            query=explanation,
            execution_info=ExecutionInfo(
                code=code,
                explanation=explanation,
                elapsed_time=0,
                console_output="",
            ),
        )

        self._upload_status(status)

        start_time = time.time()
        
        try:
            # Start async execution
            exec_task = asyncio.create_task(self.executor.run_program(code))
            
            # Poll logs frequently while code is running
            last_status_update = start_time
            while not exec_task.done():
                # Get latest logs
                logs = await self.executor.get_logs()
                elapsed = time.time() - start_time
                
                # Update execution info with current logs
                status.execution_info.console_output = logs
                status.execution_info.elapsed_time = elapsed
                
                # Upload status every 30 seconds, but check task completion every 1 second
                current_time = time.time()
                if current_time - last_status_update >= 30:
                    self._upload_status(status)
                    last_status_update = current_time
                
                # Wait 1 second before next check (much more responsive)
                await asyncio.sleep(1)
            
            # Get final exit code
            exit_code = await exec_task
            
            # Get final logs
            final_logs = await self.executor.get_logs()
            final_time = time.time() - start_time
            
            # Update status one last time
            status.execution_info.console_output = final_logs
            status.execution_info.elapsed_time = final_time
            self._upload_status(status)
            
            return self._parse_exec_result(code, explanation, final_logs, final_time)
            
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            status.execution_info.console_output = f"Error: {str(e)}"
            status.execution_info.elapsed_time = time.time() - start_time
            self._upload_status(status)
            raise

    async def launch(self):
        try:
            self._fetch_task_config()
            
            # Start heartbeat mechanism after getting task config
            self._start_heartbeat()

            exec_result = None
            for self.current_step in range(1, self.agent_config.total_steps + 1):
                if (time.time() - self.start_time) > self.agent_config.total_time_limit:
                    logger.info("Total time limit reached.")
                    break

                logger.info(f"--- Starting step {self.current_step}/{self.agent_config.total_steps} ---")

                try:
                    code, explanation = self._query_code(exec_result)
                    exec_result = await self._execute_code(code, explanation)

                except Exception as e:
                    logger.error(f"An error occurred during step {self.current_step}, execution will be terminated: {e}")
                    return

            logger.info("Maximum steps or time limit reached. Finalizing.")
            self._fetch_report()
        finally:
            # Always stop heartbeat when agent finishes
            self._stop_heartbeat()
