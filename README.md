# CoMind: Towards Community-Driven Agents for Machine Learning Engineering

[![arXiv](https://img.shields.io/badge/arXiv-2507.21184-b31b1b.svg)](https://arxiv.org/abs/2506.20640)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is the repository for the paper *CoMind: Towards Community-Driven Agents for Machine Learning Engineering*

## News

- **[2026.01.26]** 🎉 Our paper has been accepted at **ICLR 2026**!

## Running CoMind on Kaggle Competitions



## Configuration Guide

You can use your own configuration file by specifying `--config <path>`. Here is the configuration structure for CoMind:

```yaml
competition:
  id: null         # Competition ID
  input_dir: null  # Path to the dataset of the competition 
  task_desc: null  # Task description

llm:
  model: gpt-5-mini  # Model name
  max_retries: 20    # Retry attempts
  timeout: 600       # API timeout (seconds)

agent:
  num_code_agents: 2             # Number of parallel code agents
  num_iterations: 20             # Maximum iteration steps for CoMind
  num_iterations_code_agents: 30 # Maximum iteration steps for code agents
  time_limit_code_agents: 18000  # Time constraint (seconds) for code agents

  max_referred_discussions: 10 # Top discussions to consider
  max_referred_kernels: 1      # Top kernels to consider

  workspace_dir: workspace             # Path to the workspace folder
  external_data_dir: datasets          # Path to the external datasets folder
  submission_file_name: submission.csv # Expected submission file name
  separate_validation_submission: True # Wether to enable the Evaluator
  reproduce: False                     # Wether to reproduce shared kernels

execution:
  inspect_interval: 1200 # Interval (seconds) for execution inspection
  timeout: 18000         # Time limit (seconds) for a single execution
  max_cpu_cores: 24      # CPU constraint for a single execution
  max_gpu_count: 1       # GPU constraint for a single execution#
``` 