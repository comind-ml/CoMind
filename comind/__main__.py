"""
Module runner for comind monitoring panel.
Usage: python -m comind.monitor
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="ComInd Monitoring Panel")
    parser.add_argument("--config", type=str, default="comind.yaml", help="Path to the config file")
    parser.add_argument("--competition_id", "-c", type=str, default=None, help="Competition ID")
    parser.add_argument("--data-dir", "-d", type=Path, default=None, help="Path to the competition input directory")
    parser.add_argument("--task-desc", "-t", type=str, default=None, help="Path to the competition task description")
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from comind.config.config import Config
    from comind.agent import Agent
    from comind.monitor import run_monitor
    
    # Load config
    config = Config.from_yaml(args.config)
    
    if args.competition_id is not None:
        config.competition_id = args.competition_id
    
    if args.data_dir is not None:
        config.competition_input_dir = args.data_dir
    
    if config.competition_id is None or config.competition_input_dir is None:
        print("Error: You must specify both competition ID and data directory")
        print("Usage: python -m comind.monitor -c <competition_id> -d <data_dir>")
        sys.exit(1)
    
    if args.task_desc is not None:
        config.competition_task_desc = args.task_desc
    elif config.competition_task_desc is None:
        print(f"Warning: No task description provided. Trying to read {config.competition_input_dir}/description.md")
        config.competition_task_desc = config.competition_input_dir / "description.md"
    
    try:
        with open(config.competition_input_dir / "description.md", "r", encoding="utf-8") as f:
            config.competition_task_desc = f.read()
    except FileNotFoundError:
        print(f"Warning: Could not read task description from {config.competition_input_dir}/description.md")
        config.competition_task_desc = "No task description available."
    
    # For monitoring, we don't need to create a new workspace directory
    # We'll use existing data if available
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.agent_workspace_dir = config.agent_workspace_dir / config.competition_id / timestamp
    
    print("Initializing agent for monitoring...")
    try:
        agent = Agent(config)
        print(f"Agent initialized with {len(agent.reports)} reports, {len(agent.ideas)} ideas, and {len(agent.datasets)} datasets.")
        run_monitor(agent)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

