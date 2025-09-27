import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

log_dir_path = os.path.join(PROJECT_ROOT, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

# Create logger globally
logger = logging.getLogger("AutoSenseLogger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

# File handler
file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
