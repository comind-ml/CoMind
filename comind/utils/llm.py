import litellm
import re
from comind.config import LLMConfig
from typing import Callable

def extract_fields(response: str, fields: list[str]) -> dict[str, list[str]]:
    result = {}
    for field in fields:
        pattern = f"<{field}>([^<]*(?:<(?!/{field}>)[^<]*)*)</{field}>"
        matches = re.findall(pattern, response) 
        if len(matches) == 0:
            raise ValueError(f"Field {field} not found in response.")
        result[field] = [match.strip() for match in matches]
    return result
        
def query_llm_raw(cfg: LLMConfig, messages: list[dict]) -> str:
    for _ in range(cfg.max_retries):
        try:
            response = litellm.completion(model=cfg.model, messages=messages, **cfg.params)
            print(response)
            if not response.choices[0].finish_reason == "stop":
                continue 
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {e}")
            continue

    raise ValueError("Failed to get a valid response from the LLM.")

def query_llm(cfg: LLMConfig, messages: list[dict], required_fields: list[str] = [], check_fn: Callable[[dict], bool] = None) -> dict:
    for _ in range(cfg.max_retries):
        try:
            response = query_llm_raw(cfg, messages)
            fields = extract_fields(response, required_fields)
            if check_fn:
                if not check_fn(fields):
                    continue
            # Add raw content to the result
            fields['_raw_content'] = response
            return fields
        except Exception as e:
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            print(f"Error querying LLM: {e}")
            continue
    raise ValueError("Failed to get a valid response from the LLM.")

class Conversation:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def pop_message(self):
        return self.messages.pop()
    
    def query(self, required_fields: list[str] = [], check_fn: Callable[[dict], bool] = None) -> dict:
        return query_llm(self.cfg, self.messages, required_fields, check_fn)