import logging
import os
from src import config

# Create the logs directory in the repository root if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Set up a logger to log detailed info to a file while printing only warnings/errors to the console.
logger = logging.getLogger("ProbeLogger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "main.log"))
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_info(message):
    logger.info(message)

def get_probe_output_dir(dataset, model, task, probe_type, *, pca=False,
                         pca_dim=None, base_dir=None):
    if base_dir is None:
        base_dir = config.OUTPUT_DIR
    name = f"{dataset}_{model}_{task}_{probe_type}"
    if pca and pca_dim:
        name += f"_pca{pca_dim}"
    return os.path.join(base_dir, name)
