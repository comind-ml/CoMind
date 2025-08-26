import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses_json import DataClassJsonMixin
import time

@dataclass
class LLMConfig:
    model: str = "o4-mini"  
    params: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3

    def __post_init__(self):
        # Ensure params is a dict if None was provided
        if self.params is None:
            self.params = {}

@dataclass
class Config(DataClassJsonMixin):
    # Competition settings
    competition_id: Optional[str] = None
    competition_input_dir: Optional[Path] = None
    competition_task_desc: Optional[str] = None

    # LLM settings
    llm: LLMConfig = field(default_factory=lambda: LLMConfig())

    # Agent settings 
    agent_num_code_agents: int = 4
    agent_num_iterations: int = 10
    agent_num_iterations_code_agents: int = 20
    agent_time_limit_code_agents: int = 18000
    agent_max_referred_discussions: int = 10
    agent_max_referred_kernels: int = 10
    agent_workspace_dir: Path = Path("workspace")
    agent_external_data_dir: Optional[Path] = None

    # Execution settings
    execution_inspect_interval: int = 1200
    execution_timeout: int = 18000
    execution_max_cpu_cores: int = 8
    execution_max_gpu_count: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
            
            # Handle LLM section specially
            llm_section = config_dict.get("llm", {})
            llm_config = LLMConfig(
                model=llm_section.get("model", "o4-mini"),
                max_retries=llm_section.get("max_retries", 3)
            )
            # Any additional parameters in llm section become params
            llm_params = {k: v for k, v in llm_section.items() if k not in ["model", "max_retries"]}
            llm_config.params = llm_params
            
            # Transform competition, agent, and execution sections
            flat_dict = {}
            for section in ["competition", "agent", "execution"]:
                if section in config_dict:
                    for key, value in config_dict[section].items():
                        # Convert string paths to Path objects
                        if key in ["input_dir", "workspace_dir", "external_data_dir"] and value is not None:
                            value = Path(value)
                        flat_dict[f"{section}_{key}"] = value
            
            # Create config instance
            config = cls.from_dict(flat_dict)
            config.llm = llm_config

            config.start_time = time.time()
            return config

    def to_yaml(self, yaml_path: Path) -> None:
        # Transform flat structure back to nested structure
        nested_dict = {
            "competition": {
                "id": self.competition_id,
                "input_dir": str(self.competition_input_dir) if self.competition_input_dir else None,
                "task_desc": self.competition_task_desc
            },
            "llm": {
                "model": self.llm.model,
                "max_retries": self.llm.max_retries,
                **self.llm.params  # Include all additional parameters
            },
            "agent": {
                "num_code_agents": self.agent_num_code_agents,
                "num_iterations": self.agent_num_iterations,
                "num_iterations_code_agents": self.agent_num_iterations_code_agents,
                "time_limit_code_agents": self.agent_time_limit_code_agents,
                "max_referred_discussions": self.agent_max_referred_discussions,
                "max_referred_kernels": self.agent_max_referred_kernels,
                "workspace_dir": str(self.agent_workspace_dir) if self.agent_workspace_dir else None,
                "external_data_dir": str(self.agent_external_data_dir) if self.agent_external_data_dir else None
            },
            "execution": {
                "inspect_interval": self.execution_inspect_interval,
                "timeout": self.execution_timeout,
                "max_cpu_cores": self.execution_max_cpu_cores,
                "max_gpu_count": self.execution_max_gpu_count
            }
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(nested_dict, f, default_flow_style=False)


