import re

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
