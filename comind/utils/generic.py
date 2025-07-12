from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from functools import wraps
import time
from typing import List, Dict, Any

def get_timestamp() -> str:
    """Get the current timestamp """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def read_file_content(path: Path) -> str:
    """Reads the content of a file."""
    if path.is_file():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"File not found: {path}")

def get_kernel_code_files(kernels_path: Path) -> List[Path]:
    """Finds all Python/IPython notebook files in the kernels directory."""
    if not kernels_path.is_dir():
        return []
    
    # Each kernel is in its own subdirectory
    code_files = []
    for kernel_dir in kernels_path.iterdir():
        if kernel_dir.is_dir():
            # Find .py or .ipynb files
            py_files = list(kernel_dir.glob('*.py'))
            ipynb_files = list(kernel_dir.glob('*.ipynb'))
            if py_files:
                code_files.append(py_files[0])
            elif ipynb_files:
                code_files.append(ipynb_files[0])
    return code_files

def get_discussion_files(discussions_path: Path) -> List[Path]:
    """Finds all markdown files in the discussions directory."""
    if not discussions_path.is_dir():
        return []
    return list(discussions_path.glob('**/*.md'))

def _read_artifacts_from_path(path: Path) -> List[Dict[str, Any]]:
    """Helper function to read artifacts based on an info.json file."""
    if not path or not path.is_dir():
        return []

    info_path = path / "info.json"
    if not info_path.exists():
        return []

    try:
        info_list = json.loads(read_file_content(info_path))
        if not isinstance(info_list, dict):
            return []

        artifacts = []
        for id, item in info_list.items():
            content_path = path / f"{id}.txt"
            content = read_file_content(content_path)
            if content:
                item_with_content = item.copy()
                item_with_content["content"] = content
                artifacts.append(item_with_content)
        return artifacts
    except (json.JSONDecodeError, TypeError):
        return []

def read_discussions_from_path(discussions_path: Path) -> List[Dict[str, Any]]:
    """Reads all discussion files based on info.json in the given path."""
    return _read_artifacts_from_path(discussions_path)

def read_kernels_from_path(kernels_path: Path) -> List[Dict[str, Any]]:
    """Reads all kernel files based on info.json in the given path."""
    return _read_artifacts_from_path(kernels_path)