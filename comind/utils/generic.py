from datetime import datetime
import json
from pathlib import Path
import pandas as pd

def get_timestamp() -> str:
    """Get the current timestamp """
    return datetime.now().strftime("%Y-%m-%d")

def str_to_datetime(date_str: str) -> datetime:
    """Convert a string to a datetime object"""
    return pd.to_datetime(date_str).to_pydatetime()

def read_raw_markdown(file_path: Path) -> str:
    with open(file_path, "r", encoding='utf-8') as f:
        return "```\n" + f.read() + "\n```\n"
    
def read_jupyter_notebook(file_path: Path) -> str:
    """ Read the code and markdown cells from a Jupyter notebook. """

    assert file_path.suffix == ".ipynb", "File must be a Jupyter notebook."

    with open(file_path, "r", encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            """ This indicates that the file is a raw markdown file. """
            return read_raw_markdown(file_path)

    result = ""
    last_cell_type = "markdown"
    for cell in notebook["cells"]:
        if last_cell_type != cell["cell_type"]:
            result += "\n```\n"
        else:
            result += "\n"
        if cell["cell_type"] == "code":
            result += "".join(cell["source"])
        elif cell["cell_type"] == "markdown":
            result += "".join(cell["source"])
        last_cell_type = cell["cell_type"]
    if last_cell_type == "code":
        result += "\n```\n"
    return result