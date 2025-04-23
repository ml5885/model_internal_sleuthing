import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    # Original models
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
    
    # Encoder‑only masked‑LMs
    "bert-base-uncased": {
        "model_name": "google-bert/bert-base-uncased",
        "tokenizer_name": "google-bert/bert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "bert-large-uncased": {
        "model_name": "bert-large-uncased",
        "tokenizer_name": "bert-large-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "distilbert-base-uncased": {
        "model_name": "distilbert/distilbert-base-uncased",
        "tokenizer_name": "distilbert/distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "deberta-v3-large": {
        "model_name": "microsoft/deberta-v3-large",
        "tokenizer_name": "microsoft/deberta-v3-large",
        "max_length": 128,
        "batch_size": 32,
    },

    # DeepSeek reasoning‑distilled models
    "deepseek-r1-distill-qwen-7b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "trust_remote_code": True,
        "max_length": 128,
        "batch_size": 32,
    },
    "deepseek-r1-distill-llama-8b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "trust_remote_code": True,
        "max_length": 128,
        "batch_size": 32,
    },

    # Meta Llama models
    "llama2-7b-chat": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "tokenizer_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_length": 128,
        "batch_size": 32,
    },
    "llama3-1b-instruct": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.2-1B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },

    # Additional causal LMs
    "gpt-neo-2.7b": {
        "model_name": "EleutherAI/gpt-neo-2.7B",
        "tokenizer_name": "EleutherAI/gpt-neo-2.7B",
        "max_length": 128,
        "batch_size": 32,
    },
    "opt-2.7b": {
        "model_name": "facebook/opt-2.7b",
        "tokenizer_name": "facebook/opt-2.7b",
        "max_length": 128,
        "batch_size": 32,
    },
    "pythia2.8b": {
        "model_name": "EleutherAI/pythia-2.8b-v0",
        "tokenizer_name": "EleutherAI/pythia-2.8b-v0",
        "max_length": 128,
        "batch_size": 32,
    }
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

SEED = 42

CLUSTERING = {
    "n_clusters": 2,
    "random_state": SEED
}

MAX_WORKERS_CLASS = 2
