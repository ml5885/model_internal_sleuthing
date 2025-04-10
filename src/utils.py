import logging
import os

# Configure logging format and level
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    logging.info(message)

def log_debug(message, data=None):
    logging.debug(message)
    if data is not None:
        logging.debug(str(data))

def log_error(message):
    logging.error(message)

def read_csv(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def save_npz(filepath, **kwargs):
    import numpy as np
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **kwargs)
