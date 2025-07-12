from dataclasses import dataclass

@dataclass
class Registry:
    code: str = None
    desc: str = None
    pipeline: str = None
    gpu_info: str = None
    data_overview: str = None
    output: str = None
    remain_steps: int = None
    remain_time: float = None
    execution_review: str = None
    explanation: str = None
    time_limit_per_step: int = None
    total_time_limit: int = None

alias = {
    "code": "CODE", 
    "desc": "DESC",
    "pipeline": "PIPELINE",
    "gpu_info": "GPU",
    "data_overview": "DATA_OVERVIEW",
    "output": "OUTPUT",
    "remain_steps": "REMAIN_STEPS",
    "remain_time": "REMAIN_TIME",
    "execution_review": "EXECUTION_REVIEW",
    "explanation": "EXPLANATION",
    "time_limit_per_step": "TIME_LIMIT",
    "total_time_limit": "TOTAL_TIME_LIMIT",
}

def embed(prompt: str, registry: Registry) -> str:
    """
    Embed the registry into the prompt.
    """
    for key, value in registry.__dict__.items():
        if value is not None:
            # Convert value to string to handle int, float, and other types
            prompt = prompt.replace(f"!<{alias[key]}>!", str(value))

    if prompt.find("!<") != -1:
        raise ValueError("Prompt contains unembedded variables")
    return prompt
