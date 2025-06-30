import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
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
    "deberta-v3-large": {
        "model_name": "microsoft/deberta-v3-large",
        "tokenizer_name": "microsoft/deberta-v3-large",
        "max_length": 128,
        "batch_size": 32,
    },
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
    "olmo2-7b": {
        "model_name": "allenai/OLMo-2-1124-7B",
        "tokenizer_name": "allenai/OLMo-2-1124-7B",
        "max_length": 128,
        "batch_size": 32,
        "checkpoints": [
            "main",
            "stage1-step5000-tokens21B",
            "stage1-step40000-tokens168B",
            "stage1-step97000-tokens407B",
            "stage1-step179000-tokens751B",
            "stage1-step282000-tokens1183B",
            "stage1-step409000-tokens1716B",
            "stage1-step559000-tokens2345B",
            "stage1-step734000-tokens3079B",
            "stage1-step928646-tokens3896B"
        ]
    },
        "pythia-6.9b": {
        "model_name": "EleutherAI/pythia-6.9b",
        "tokenizer_name": "EleutherAI/pythia-6.9b",
        "max_length": 128,
        "batch_size": 32,
        "checkpoints": [
            "main",
            "step1",
            "step64",
            "step6000",
            "step19000",
            "step37000",
            "step57000",
            "step82000",
            "step111000",
            "step143000"
        ],
    },
    "pythia-6.9b-tulu": {
        "model_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "tokenizer_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "max_length": 128,
        "batch_size": 32,
    },
    "olmo2-7b-instruct": {
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "tokenizer_name": "allenai/OLMo-2-1124-7B-Instruct",
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
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
        "max_length": 128,
        "batch_size": 32,
    },
    "gemma2b-it": {
        "model_name": "google/gemma-2-2b-it",
        "tokenizer_name": "google/gemma-2-2b-it",
        "max_length": 128,
        "batch_size": 32,
    },
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
    "xglm-2.9B": {
        "model_name": "facebook/xglm-2.9B",
        "tokenizer_name": "facebook/xglm-2.9B",
        "max_length": 128,
        "batch_size": 32,
    },
    "aya-23-8b": {
        "model_name": "CohereLabs/aya-23-8B",
        "tokenizer_name": "CohereLabs/aya-23-8B",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2.5-7B": {
        "model_name": "Qwen/Qwen2.5-7B",
        "tokenizer_name": "Qwen/Qwen2.5-7B",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2.5-7B-instruct": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_eng_latn_1000mb": {
        "model_name": "goldfish-models/eng_latn_1000mb",
        "tokenizer_name": "goldfish-models/eng_latn_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_zho_hans_1000mb": {
        "model_name": "goldfish-models/zho_hans_1000mb",
        "tokenizer_name": "goldfish-models/zho_hans_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_deu_latn_1000mb": {
        "model_name": "goldfish-models/deu_latn_1000mb",
        "tokenizer_name": "goldfish-models/deu_latn_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_fra_latn_1000mb": {
        "model_name": "goldfish-models/fra_latn_1000mb",
        "tokenizer_name": "goldfish-models/fra_latn_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_rus_cyrl_1000mb": {
        "model_name": "goldfish-models/rus_cyrl_1000mb",
        "tokenizer_name": "goldfish-models/rus_cyrl_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "goldfish_tur_latn_1000mb": {
        "model_name": "goldfish-models/tur_latn_1000mb",
        "tokenizer_name": "goldfish-models/tur_latn_1000mb",
        "max_length": 128,
        "batch_size": 32,
    },
    "byt5": {
        "model_name": "google/byt5-base",
        "tokenizer_name": "google/byt5-base",
        "max_length": 128,
        "batch_size": 32,
        "n_layers": 12,
    },
    "mt5": {
        "model_name": "google/mt5-base",
        "tokenizer_name": "google/mt5-base",
        "max_length": 128,
        "batch_size": 32,
        "n_layers": 12,
    },
}

MODEL_DISPLAY_NAMES = {
    # Decoder/causal LMs
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "xglm-2.9B": "XGLM-2.9B",
    "aya-23-8b": "Aya-23-8B",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    # Encoder-only masked LMs
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "distilbert-base-uncased": "DistilBERT-Base",
    "deberta-v3-large": "DeBERTa-v3-Large",
    # Meta Llama
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    # HuggingFace/other names for analogy completion
    "google-bert/bert-base-uncased": "BERT-Base",
    "microsoft/deberta-v3-large": "DeBERTa-v3-Large",
    "openai-community/gpt2-large": "GPT-2-Large",
    "openai-community/gpt2-xl": "GPT-2-XL",
    "EleutherAI/pythia-6.9b": "Pythia-6.9B",
    "allenai/open-instruct-pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-7B": "OLMo-2-1124-7B",
    "google/gemma-2-2b": "Gemma-2-2B",
    "google/gemma-2-2b-it": "Gemma-2-2B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.1-8B": "Llama-3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3-8B-Instruct",
}

ANALOGY_MODEL_LIST = [
    "google-bert/bert-base-uncased", "bert-large-uncased", "microsoft/deberta-v3-large",
    "gpt2", "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "EleutherAI/pythia-6.9b", "allenai/open-instruct-pythia-6.9b-tulu",
    "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-1124-7B",
    "google/gemma-2-2b", "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "qwen/Qwen2.5-7B", "qwen/Qwen2.5-7B-Instruct",
]

MODEL_TABLE_MAPPING = {
    'BERT-Base (768)': 'bert-base-uncased',
    'BERT-Large (1024)': 'bert-large-uncased',
    'DeBERTa-v3-Large (1024)': 'deberta-v3-large',
    'GPT-2-Small (768)': 'gpt2',
    'GPT-2-Large (1280)': 'gpt2-large',
    'GPT-2-XL (1600)': 'gpt2-xl',
    'Pythia-6.9B (4096)': 'pythia-6.9b',
    'Pythia-6.9B-Tulu (4096)': 'pythia-6.9b-tulu',
    'OLMo-2-7B (4096)': 'olmo2-7b',
    'OLMo-2-7B-Instruct (4096)': 'olmo2-7b-instruct',
    'Gemma-2-2B (2304)': 'gemma2b',
    'Gemma-2-2B-Instruct (2304)': 'gemma2b-it',
    'Qwen-2.5-1.5B (1536)': 'qwen2',
    'Qwen-2.5-1.5B-Instruct (1536)': 'qwen2-instruct',
    'Llama-3.1-8B (4096)': 'llama3-8b',
    'Llama-3.1-8B-Instruct (4096)': 'llama3-8b-instruct',
}

# Training hyperparameters for probing
TRAIN_PARAMS = {
    "epochs": 50,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "batch_size": 4096,
    "early_stop": 7,
    
    # For RF probes
    "rf_n_estimators": 10,
    "rf_max_depth": 10,
    "rf_min_samples_leaf": 10,
    # use all but one of your allocated cores
    "workers": max(1, os.cpu_count() - 1),
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