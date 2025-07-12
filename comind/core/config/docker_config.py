from pydantic import BaseModel, Field
from typing import List, Optional

class DockerConfig(BaseModel):
    image: str = Field(..., description="Docker image name")
    mounts: List[str] = Field(default_factory=list, description="Host-container mount relationships, e.g. ['/host/path:/container/path']")
    code_path: str = Field(..., description="Path to save code inside the container")
    cpu: Optional[List[int]] = Field(None, description="List of assigned CPU core indices")
    gpu: Optional[str] = Field(None, description="Assigned GPU, e.g. 'all' or '0'")
    timeout: int = Field(3600, description="Timeout for code execution in seconds")
