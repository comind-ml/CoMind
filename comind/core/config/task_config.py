from pydantic import BaseModel, Field

class TaskConfig(BaseModel):
    agent_id: str = Field(..., description="The agent ID")
    task_desc: str = Field(..., description="The task description")
    pipeline: str = Field(..., description="The implementation draft")
    data_overview: str = Field(..., description="The data overview of input files")

