import asyncio
import time
import uuid

from pathlib import Path
from typing import Any, Dict

from comind.core import logger
from comind.core.agent.status import AgentState, ExecutionInfo, Status
from comind.core.assets.registry import Registry, embed
from comind.core.agent.metric import MetricValue
from comind.core.config import AgentConfig, DockerConfig, LLMConfig
from comind.environment.docker import Docker
from comind.llm.llm import LLM
from comind.core.assets.fn_specs import review_func_spec

import requests

class Agent:
    def __init__(
        self, 
        llm_config: LLMConfig,
        docker_config: DockerConfig,
    ):
        self.agent_id = uuid.uuid4()
        self.agent_config = None
        self.llm_config = llm_config
        self.docker_config = docker_config

        self.executor = Docker(docker_config)
        self.llm = LLM(llm_config)
        self.feedback_llm = LLM(llm_config, keep_history=False)

    def _post_request(self, url: str, data: Dict[str, Any]):
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"Failed to post request to {url}: {response.status_code} {response.text}")

        return response.json()
    
    def _fetch_agent_config(self):
        assert self.agent_config is None 
        request_url = self.agent_config.api_url + "/config"
        data = {
            "agent_id": self.agent_id,
        }
        response = self._post_request(request_url, data)
        self.agent_config = AgentConfig(**response)

        if self.agent_id != self.agent_config.agent_id:
            raise Exception(f"Agent ID mismatch: Expected {self.agent_id}, got {self.agent_config.agent_id}")

    def _upload_report(self, report: Dict[str, Any]):
        request_url = self.agent_config.api_url + "/report"
        data = {
            "agent_id": self.agent_config.agent_id,
            "report": report,
        }
        response = self._post_request(request_url, data)
        return response

    def _upload_status(self):
        pass 

    def _upload_submission(self, file_path: Path, metric: MetricValue):
        pass

    def _parse_exec_result(self, code: str, explanation: str, output: str, exec_time: float) -> str:
        info = Registry(
            code=code,
            output=output + f"\n\n[System] Execution time: {exec_time:.2f}s",
            explanation=explanation,
        )

        prompt = embed(self.agent_config.pipeline, info)
        feedback = self.feedback_llm.chat(prompt, function_spec=review_func_spec)

        metric = feedback["metric"] 
        
    
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
            
            # Poll logs every 30 seconds while code is running
            while not exec_task.done():
                # Get latest logs
                logs = await self.executor.get_logs()
                elapsed = time.time() - start_time
                
                # Update execution info with current logs
                status.execution_info.console_output = logs
                status.execution_info.elapsed_time = elapsed
                
                # Upload updated status
                self._upload_status(status)
                
                # Wait 30 seconds before next poll
                await asyncio.sleep(30)
            
            # Get final exit code
            exit_code = await exec_task
            
            # Get final logs
            final_logs = await self.docker.get_logs()
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

