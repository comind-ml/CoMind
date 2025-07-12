import re
import json
from typing import Optional, Dict, Any

def trim_long_string(string: str, threshold: int = 10010, k: int = 5000) -> str:
    """
    Trim a string to a maximum length of threshold characters, 
    keeping k characters from the beginning and end.
    """
    if len(string) <= threshold:
        return string
    
    truncated_len = len(string) - k * 2
    return f"{string[:k]}\n ... [{truncated_len} characters truncated] ... \n{string[-k:]}"

def extract_code(text: str) -> str:
    """Extract code block from a string"""
    matches = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)

    if not matches:
        return None
    
    return matches[0]

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a string.
    It first looks for a JSON object inside a markdown code block.
    If that fails, it falls back to finding the first substring that is a valid JSON.
    """
    # 1. Try to find a JSON markdown code block
    match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # The content of the markdown block is not valid JSON,
            # so we continue to the next method.
            pass
    
    # 2. Fallback to finding the first valid JSON object in the text
    json_pattern = re.compile(r'{.*?}', re.DOTALL)
    matches = json_pattern.findall(text)

    for potential_json in matches:
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue

    return None
