from dataclasses import Field
from pydantic import BaseModel

class AgentConfig(BaseModel):
    agent_id: str = Field(..., description="The agent ID")
    task_desc: str = Field(..., description="The task description")
    pipeline: str = Field(..., description="The implementation draft")
    data_overview: str = Field(..., description="The data overview of input files")
    api_url: str = Field(..., description="The API URL to interact with the simulated community")

