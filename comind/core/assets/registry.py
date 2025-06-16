from dataclasses import dataclass

@dataclass
class Registry:
    code: str = None
    output: str = None
    remain_steps: int = None
    remain_time: float = None
    execution_review: str = None
    explanation: str = None

alias = {
    "code": "CODE", 
    "output": "OUTPUT",
    "remain_steps": "REMAIN_STEPS",
    "remain_time": "REMAIN_TIME",
    "execution_review": "EXECUTION_REVIEW",
    "explanation": "EXPLANATION",
}

def embed(prompt: str, registry: Registry) -> str:
    """
    Embed the registry into the prompt.
    """
    for key, value in registry.__dict__.items():
        if value is not None:
            prompt = prompt.replace(f"!<{alias[key]}>!", value)

    if prompt.find("!<") != -1:
        raise ValueError("Prompt contains unembedded variables")
    return prompt
