import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "tokenizer_name": "gpt2",
        "max_length": 128,
        "batch_size": 32,
    },
    "gpt2-xl": {
        "model_name": "openai-community/gpt2-xl",
        "tokenizer_name": "openai-community/gpt2-xl",
        "max_length": 128,
        "batch_size": 32,
    },
    "gpt2-large": {
        "model_name": "openai-community/gpt2-large",
        "tokenizer_name": "openai-community/gpt2-large",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2-instruct": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B",
        "max_length": 128,
        "batch_size": 32,
    },
    "pythia1.4b": {
        "model_name": "EleutherAI/pythia-1.4b",
        "tokenizer_name": "EleutherAI/pythia-1.4b",
        "max_length": 128,
        "batch_size": 32,
    },
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
        "max_length": 128,
        "batch_size": 32,
    },
    
    "gemma2b-it": {
        "model_name": "google/gemma-2b-it",
        "tokenizer_name": "google/gemma-2b-it",
        "max_length": 128,
        "batch_size": 32,
    },
    "olmo2-7b": {
          "model_name": "allenai/OLMo-2-1124-7B",
          "tokenizer_name": "allenai/OLMo-2-1124-7B",
          "max_length": 128,
          "batch_size": 32,
     },
     "olmo2-7b-instruct": {
          "model_name": "allenai/OLMo-2-1124-7B-Instruct",
          "tokenizer_name": "allenai/OLMo-2-1124-7B-Instruct",
          "max_length": 128,
          "batch_size": 32,
     },
    
    # Encoder-only masked-LMs
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
    
    # Meta Llama models
    "llama3-8b": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "tokenizer_name": "meta-llama/Llama-3.1-8B",
        "max_length": 128,
        "batch_size": 32,
    },
    "llama3-8b-instruct": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    
    "pythia-6.9b": {
        "model_name": "EleutherAI/pythia-6.9b",
        "tokenizer_name": "EleutherAI/pythia-6.9b",
        "max_length": 128,
        "batch_size": 32,
    },
    "pythia-6.9b-tulu": {
        "model_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "tokenizer_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "max_length": 128,
        "batch_size": 32,
    },

    # More open models
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
    },

    # Reasoning models
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
}

# Training hyperparameters for probing
TRAIN_PARAMS = {
    "epochs": 50,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "batch_size": 4096,
    "early_stop": 7,
    "workers": min(4, os.cpu_count() or 2),
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
