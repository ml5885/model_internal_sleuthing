import logging
import os

# Create the logs directory in the repository root if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Set up a logger to log detailed info to a file while printing only warnings/errors to the console.
logger = logging.getLogger("ProbeLogger")
logger.setLevel(logging.DEBUG)  # Capture all logs; filtering will be done in handlers

# File handler: logs INFO and above to a file
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "main.log"))
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler: only show warnings and errors on console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_info(message):
    logger.info(message)

def log_debug(message, data=None):
    logger.debug(message)
    if data is not None:
        logger.debug(str(data))

def log_error(message):
    logger.error(message)

def read_csv(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def save_npz(filepath, **kwargs):
    import numpy as np
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **kwargs)
