from dataclasses import dataclass
from .dataset import Dataset

@dataclass
class Draft:
    id: str
    title: str
    description: str
    datasets: list[Dataset]
    codebase: str | None
    codebase_content: str | None
    code: str