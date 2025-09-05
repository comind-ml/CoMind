import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from comind.config import Config
from comind.utils import process_backspace_chars


@dataclass
class ExecutionResult:
    """Class to collect execution results."""
    success: bool
    output: str
    error: Optional[str] = None

def execute_file(workspace_dir: Path, file_path: Path, extra_args: list[str] = []) -> ExecutionResult:
    try:
        # Start the process
        process = subprocess.run(
            ['python', file_path.absolute()] + extra_args,
            capture_output=True,
            text=True,
            cwd=str(workspace_dir)
        )
        
        # Get the output and process backspace characters
        raw_output = process.stdout
        if process.stderr:
            raw_output += "\n" + process.stderr
            
        final_output = process_backspace_chars(raw_output)
        
        success = process.returncode == 0
        
        return ExecutionResult(
            success=success,
            output=final_output,
            error=process.stderr
        )
        
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            error=str(e)
        )

def execute(workspace_dir: Path, code: str, name: str, extra_args: list[str] = []) -> ExecutionResult:
    """Execute code and return results.
    
    Args:
        code: Python code to execute
        name: Name of the script to execute (with .py)
        extra_args: Extra arguments to pass to the script
        
    Returns:
        ExecutionResult containing execution details
    """

    agent_file = workspace_dir / f"{name}"
    agent_file.write_text(code)

    return execute_file(workspace_dir, agent_file, extra_args)
        