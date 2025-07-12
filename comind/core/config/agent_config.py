from pathlib import Path
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    agent_id: str = Field(..., description="The ID of the agent")
    api_url: str = Field(..., description="The API URL to interact with the simulated community")
    total_steps: int = Field(..., description="The total number of steps to run")
    total_time_limit: int = Field(..., description="The total time limit for the agent to run")
    time_limit_per_step: int = Field(..., description="The time limit for each step")
    working_dir: Path = Field(..., description="The working directory for the agent")