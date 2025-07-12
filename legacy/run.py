import torch
import logging
import sys
import shutil
import atexit

import backend
from rich.status import Status
from rich.tree import Tree
from utils.config import load_task_desc, prep_agent_workspace, load_cfg
from agent import Agent

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)

def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # dont want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("graphml")
    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "graphml.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "graphml.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    
    agent = Agent(
        task_desc=task_desc_str,
        cfg=cfg,
    )

    agent.run()

if __name__ == "__main__":
    run()