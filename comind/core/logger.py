from comind.utils.generic import get_timestamp

from pathlib import Path
import logging
import re
import os
import sys

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

class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode encoding issues safely"""
    
    def format(self, record: logging.LogRecord) -> str:
        try:
            # Convert message to string and handle Unicode issues
            if hasattr(record.msg, '__str__'):
                # Ensure the message can be safely encoded
                msg_str = str(record.msg)
                # Replace problematic characters that can't be encoded
                msg_str = msg_str.encode('utf-8', errors='replace').decode('utf-8')
                record.msg = msg_str
            
            formatted = super().format(record)
            # Ensure the final formatted string is safe
            return formatted.encode('utf-8', errors='replace').decode('utf-8')
        except Exception as e:
            # Fallback to a basic safe message if formatting fails
            return f"[LOGGING ERROR] Failed to format log message: {str(e)}"

class UTF8StreamHandler(logging.StreamHandler):
    """StreamHandler that forces UTF-8 encoding for console output"""
    
    def __init__(self, stream=None):
        super().__init__(stream)
        # Force UTF-8 encoding on Windows
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Ensure message can be safely written
            if hasattr(stream, 'encoding') and stream.encoding:
                # Try to encode with the stream's encoding first
                try:
                    msg.encode(stream.encoding)
                except UnicodeEncodeError:
                    # If that fails, replace problematic characters
                    msg = msg.encode(stream.encoding, errors='replace').decode(stream.encoding)
            
            stream.write(msg + self.terminator)
            self.flush()
        except Exception as e:
            # Fallback error handling
            try:
                fallback_msg = f"[LOGGING ERROR] {str(e)}\n"
                self.stream.write(fallback_msg.encode('ascii', errors='replace').decode('ascii'))
                self.flush()
            except Exception:
                pass  # Give up if even the fallback fails

def setup_console_handler(level: int = logging.INFO) -> logging.StreamHandler:
    """Setup a console handler with UTF-8 encoding and safe formatting"""
    handler = UTF8StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = SafeFormatter(CONSOLE_FMT)
    handler.setFormatter(formatter)
    return handler

def setup_file_handler(log_dir: Path, level: int = logging.INFO) -> logging.FileHandler:
    """Setup a file handler with UTF-8 encoding and safe formatting"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    log_file = log_dir / f"comind_{timestamp}.log"
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(level)
    formatter = SafeFormatter(FILE_FMT)
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
