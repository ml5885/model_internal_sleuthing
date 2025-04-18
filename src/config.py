import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configurations: update model names based on Hugging Face repository information.
MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "tokenizer_name": "gpt2",
        "max_length": 128,
        "batch_size": 32,
    },
    "pythia1.4b": {
        "model_name": "EleutherAI/pythia-1.4b-v0",
        "tokenizer_name": "EleutherAI/pythia-1.4b-v0",
        "max_length": 128,
        "batch_size": 32,
    },
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
}

# Training hyperparameters for probing
TRAIN_PARAMS = {
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}

# Data split ratios
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.1,
    "test": 0.2
}

# Random seed for reproducibility
SEED = 42

# Clustering analysis settings
CLUSTERING = {
    "n_clusters": 2,  # Adjust as needed
    "random_state": SEED
}

MAX_WORKERS_CLASS = 2
