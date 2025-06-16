# CoMind: Community-Driven MLE Agent

A community-augmented LLM agent for automated machine learning engineering that leverages collective knowledge from Kaggle-style competitions.

## Installation

```bash
git clone https://github.com/your-username/comind.git
cd comind
pip install -e .
```

## Quick Start

1. **Configure the agent**:
```python
from comind.core.config import LLMConfig, DockerConfig
from comind.core.agent import Agent

llm_config = LLMConfig(
    model_name="gpt-4o-mini",
    api_key="your-api-key"
)

docker_config = DockerConfig(
    image="python:3.9",
    mounts=["/host/data:/container/input"],
    code_path="/container/code.py"
)

agent = Agent(llm_config, docker_config)
```

2. **Download competition data**:
```bash
python competition/download.py -c competition-name -d -k -t
```
- `-d`: Dataset files
- `-k`: Public kernels  
- `-t`: Discussion topics

3. **Run the agent**:
```python
# Agent will iteratively generate and execute code
# Results saved to ./working/submission.csv
```

## Key Components

- `comind/core/agent/`: Main agent implementation with 4-stage workflow
- `comind/llm/`: LLM interface with function calling support
- `comind/environment/`: Docker execution environment
- `competition/download.py`: Kaggle data collection utilities

## Configuration

Required configs:
- **LLMConfig**: Model settings, API keys, parameters
- **DockerConfig**: Container image, mounts, resource limits
- **AgentConfig**: Task description, pipeline, data overview

## License

MIT
