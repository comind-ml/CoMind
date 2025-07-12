from pydantic import SecretStr
from comind.core.agent.agent import Agent
from comind.core.config.agent_config import AgentConfig
from comind.core.config.docker_config import DockerConfig
from comind.core.config.llm_config import LLMConfig
from omegaconf import OmegaConf
import uuid
from pathlib import Path
from comind.core.logger import logger
import time
import asyncio

def main():
    config = OmegaConf.load("configs/agent/default.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)

    if config["llm"]["api_key"] == "unspecified":
        raise ValueError("API key is not provided. Please use llm.api_key=<your_api_key> to specify your API key.")

    agent_id = str(uuid.uuid4())
    working_dir = Path(config["agent"]["working_dir"]) / agent_id
    working_dir.mkdir(parents=True, exist_ok=True)

    agent_config = AgentConfig(
        agent_id            = agent_id,
        api_url             = config["agent"]["api_url"],
        total_steps         = config["agent"]["total_steps"],
        total_time_limit    = config["agent"]["total_time_limit"],
        time_limit_per_step = config["agent"]["time_limit_per_step"],
        working_dir         = working_dir,
    )

    llm_config = LLMConfig(
        model_name          = config["llm"]["model_name"],
        temperature         = config["llm"]["temperature"],
        max_tokens          = config["llm"]["max_tokens"],
        timeout             = config["llm"]["timeout"],
        native_function_calling = config["llm"]["native_function_calling"],
        keep_history        = config["llm"]["keep_history"],
        api_key             = SecretStr(config["llm"]["api_key"]),
    )

    docker_config = DockerConfig(
        image               = config["docker"]["image"],
        gpu                 = config["docker"]["gpu"],
        mounts              = [
                                f"{working_dir.absolute()}:/workspace",
                                f"{config['input_dir']}:/workspace/input",
                            ],
        timeout             = config["agent"]["time_limit_per_step"],
        code_path           = "/workspace/main.py",
    )

    while True:
        try:
            agent = Agent(agent_config, llm_config, docker_config)
            asyncio.run(agent.launch())
        except Exception as e:
            logger.error(f"Error launching agent: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()