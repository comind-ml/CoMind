from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import json

class AgentState(Enum):
    """Enum representing possible states of an agent"""
    EXECUTING = "executing"  # Executing code
    QUERYING_LLM = "querying_llm"  # Querying the LLM
    ANALYZING = "analyzing"  # Analyzing execution output

@dataclass
class ExecutionInfo:
    """Information about code execution"""
    code: str
    explanation: str
    elapsed_time: float
    console_output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "explanation": self.explanation,
            "elapsed_time": self.elapsed_time,
            "console_output": self.console_output
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionInfo':
        return cls(
            code=data["code"],
            explanation=data["explanation"],
            elapsed_time=data["elapsed_time"],
            console_output=data["console_output"]
        )

@dataclass
class Status:
    """Represents the current status of an agent"""
    state: AgentState
    query: Optional[str] = None
    execution_info: Optional[ExecutionInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "state": self.state.value
        }
        if self.query is not None:
            data["query"] = self.query
        if self.execution_info is not None:
            data["execution_info"] = self.execution_info.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Status':
        state = AgentState(data["state"])
        query = data.get("query")
        execution_info = None
        if "execution_info" in data:
            execution_info = ExecutionInfo.from_dict(data["execution_info"])
        return cls(state=state, query=query, execution_info=execution_info)

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'Status':
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
