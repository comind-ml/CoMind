import logging
import re
from pathlib import Path

class ANSIFilter(logging.Filter):
    """Filter out ANSI escape sequences from log messages."""
    def __init__(self):
        super().__init__()
        self.ansi_pattern = re.compile(r'[^\x20-\x7E\r\n\t]')
    
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.ansi_pattern.sub('', record.msg)

        if hasattr(record, 'args') and record.args:
            cleaned_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    cleaned_args.append(self.ansi_pattern.sub(self.replace_with, arg))
                else:
                    cleaned_args.append(arg)
            record.args = tuple(cleaned_args)
        
        return True

def get_logger(name: str, log_path: Path, level = logging.INFO, file_only: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s') 
        file_handler.setFormatter(formatter)

        ansi_filter = ANSIFilter()
        file_handler.addFilter(ansi_filter)
        logger.addHandler(file_handler)

        if not file_only:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger
