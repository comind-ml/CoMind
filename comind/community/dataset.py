from dataclasses import dataclass
from pathlib import Path

@dataclass
class Dataset:
    id: str
    name: str
    description: str
    base_path: Path

    def __post_init__(self):
        assert self.base_path.exists(), f"Dataset {self.base_path} does not exist"
        assert self.base_path.is_dir(), f"Dataset {self.base_path} is not a directory"
    
    def __str__(self):
        return f"<dataset>id: {self.id}, name: {self.name}</dataset>"