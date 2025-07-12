from pathlib import Path
from pydantic import SecretStr
from omegaconf import OmegaConf

from comind.core.config.comm_config import CommConfig
from comind.core.config.llm_config import LLMConfig
from comind.core.community.server import CommunityServer
from comind.core.logger import logger

def main():
    config = OmegaConf.load("configs/community/default.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)

    if config["llm"]["api_key"] == "unspecified":
        logger.error("API key is not provided. Please use 'llm.api_key=<your_api_key>' to specify your API key.")
        raise ValueError("API key is not provided.")

    comp_dir = Path(config["community"]["competition_dir"])
    data_dir = comp_dir / "input"
    desc_path = comp_dir / "description.md"
    kernels_path = comp_dir / "kernels"
    discussions_path = comp_dir / "discussions"

    comm_config = CommConfig(
        problem_desc_path       = str(desc_path),
        kernels_path            = str(kernels_path),
        discussions_path        = str(discussions_path),
        dataset_path            = str(data_dir),
        host                    = config["community"]["host"],
        port                    = config["community"]["port"],
    )

    llm_config = LLMConfig(
        model_name              = config["llm"]["model_name"],
        api_key                 = SecretStr(config["llm"]["api_key"]),
        temperature             = config["llm"]["temperature"],
        max_tokens              = config["llm"]["max_tokens"],
        timeout                 = config["llm"]["timeout"],
        native_function_calling = config["llm"]["native_function_calling"],
        keep_history            = config["llm"]["keep_history"],
    )

    logger.info("Initializing Community Server...")
    server = CommunityServer(comm_config, llm_config)
    
    try:
        logger.info("Community Server is launching...")
        server.serve()
    except Exception as e:
        logger.error(f"Community Server failed with an error: {e}")
        server.dashboard.stop() # Ensure dashboard stops on error
        raise

if __name__ == "__main__":
    main()