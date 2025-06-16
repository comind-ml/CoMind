from utils.generic import get_timestamp

from pathlib import Path
import logging
import re
import os

CONSOLE_FMT = '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s'
FILE_FMT = '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s'

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
LOG_DIR = Path(__file__).parent.parent / 'logs'

class ColorlessFormatter(logging.Formatter):
    """Formatter to remove ANSI escape codes from log messages """
    ANSI_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def format(self, record: logging.LogRecord) -> str:
        record.msg = self.ANSI_PATTERN.sub('', str(record.msg))
        return super().format(record)

def setup_console_handler(level: int = logging.INFO) -> logging.StreamHandler:
    """Setup a console handler with a specific log level """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(CONSOLE_FMT)
    handler.setFormatter(formatter)
    return handler

def setup_file_handler(log_dir: Path, level: int = logging.INFO) -> logging.FileHandler:
    """Setup a file handler with a specific log level """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    log_file = log_dir / f"comind_{timestamp}.log"
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    formatter = logging.Formatter(FILE_FMT)
    handler.setFormatter(formatter)
    return handler

logger = logging.getLogger('comind')
log_level = logging.DEBUG if DEBUG else logging.INFO

# setup console handler
console_handler = setup_console_handler(log_level)
logger.addHandler(console_handler)

# setup file handler
file_handler = setup_file_handler(LOG_DIR, log_level)
logger.addHandler(file_handler)

# set log level
logger.setLevel(log_level)
logger.info('Logger initialized')
