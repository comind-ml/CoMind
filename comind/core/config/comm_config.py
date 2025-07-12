from pydantic import BaseModel, Field
from typing import Optional

class CommConfig(BaseModel):
    """Configuration for the Community Server."""
    problem_desc_path: str = Field(..., description="Path to the markdown file describing the competition task.")
    discussions_path: Optional[str] = Field(None, description="Path to the directory containing discussion files.")
    kernels_path: Optional[str] = Field(None, description="Path to the directory containing public code kernels.")
    dataset_path: Optional[str] = Field(None, description="Path to the dataset directory for data overview generation.")
    host: str = "127.0.0.1"
    port: int = 8000

