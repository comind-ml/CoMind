from pathlib import Path
from argparse import ArgumentParser
from comind.config.config import Config
from comind.agent import Agent
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="comind.yaml", help="Path to the config file")
    parser.add_argument("--competition_id", "-c", type=str, default=None, help="Competition ID")
    parser.add_argument("--data-dir", "-d", type=Path, default=None, help="Path to the competition input directory")
    parser.add_argument("--task-desc", "-t", type=str, default=None, help="Path to the competition task description")
    parser.add_argument("--monitor", action="store_true", help="Launch monitoring panel instead of running the agent")
    parser.add_argument("--monitor-with-agent", action="store_true", help="Run agent with live monitoring panel")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.competition_id is not None:
        config.competition_id = args.competition_id

    if args.data_dir is not None:
        config.competition_input_dir = args.data_dir

    if config.competition_id is None or config.competition_input_dir is None:
        raise ValueError("You must specify both competition ID and data directory")

    if args.task_desc is not None:
        config.competition_task_desc = args.task_desc
    elif config.competition_task_desc is None:
        print(f"Warning: No task description provided. Trying to read {config.competition_input_dir}/description.md")
        config.competition_task_desc = config.competition_input_dir / "description.md"
    
    with open(config.competition_input_dir / "description.md", "r", encoding="utf-8") as f:
        config.competition_task_desc = f.read()

    config.agent_workspace_dir = config.agent_workspace_dir / config.competition_id / timestamp()

    agent = Agent(config)

    if args.monitor:
        # Launch monitoring panel only (for existing agent data)
        agent.launch_monitor()
    elif args.monitor_with_agent:
        # Run agent with live monitoring
        agent.run(start_monitor=True)
    else:
        # Run the agent normally without monitoring
        agent.run()
