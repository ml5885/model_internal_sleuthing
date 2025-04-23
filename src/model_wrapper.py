import torch
from transformers import AutoModel, AutoTokenizer
from src import config, utils
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import json

class ModelWrapper:
    def __init__(self, model_key: str):
        if model_key not in config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        self.model_config = config.MODEL_CONFIGS[model_key]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model with hidden states enabled
        self.model = AutoModel.from_pretrained(
            self.model_config["model_name"],
            output_hidden_states=True,
            trust_remote_code=self.model_config.get("trust_remote_code", False)
        )

        # If this model repo only provides a tokenizers JSON (no .model file),
        # do a direct fast-only load via huggingface_hub + tokenizers
        if self.model_config.get("trust_remote_code", False):
            try:

                repo = self.model_config["tokenizer_name"]
                # download the two JSONs
                tok_json = hf_hub_download(repo_id=repo, filename="tokenizer.json")
                cfg_json = hf_hub_download(repo_id=repo, filename="tokenizer_config.json")
                cfg = json.load(open(cfg_json, "r", encoding="utf-8"))

                # build a tokenizers Tokenizer
                tok = Tokenizer.from_file(tok_json)
                # collect any special tokens
                special_kwargs = {}
                for key in ("unk_token","bos_token","eos_token",
                            "sep_token","pad_token","cls_token","mask_token"):
                    if key in cfg:
                        special_kwargs[key] = cfg[key]
                # ensure we have a pad_token
                special_kwargs.setdefault("pad_token", special_kwargs.get("eos_token"))

                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tok,
                    use_fast=True,
                    **special_kwargs
                )
            except Exception as e:
                raise ImportError(
                    f"Failed to load DeepSeek tokenizer directly from JSON ({e}).\n"
                    "Make sure you have huggingface_hub and tokenizers installed:\n"
                    "    pip install huggingface_hub tokenizers"
                )
        else:
            # fallback to standard HF AutoTokenizer logic
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config["tokenizer_name"],
                    add_prefix_space=True,
                    use_fast=True,
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
            except Exception as fast_err:
                utils.log_info(
                    f"Fast tokenizer load failed for '{model_key}' ({fast_err}); "
                    "falling back to slow Python tokenizer."
                )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config["tokenizer_name"],
                        add_prefix_space=True,
                        use_fast=False,
                        trust_remote_code=self.model_config.get("trust_remote_code", False)
                    )
                except Exception as slow_err:
                    error_msg = (
                        f"Cannot load a tokenizer for '{model_key}'.\n"
                        "Please install the following dependencies and retry:\n"
                        "    pip install tiktoken protobuf sentencepiece"
                    )
                    utils.log_info(f"Slow tokenizer load failed for '{model_key}' ({slow_err}).")
                    raise ImportError(error_msg)

        # Ensure a pad token is defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, sentences):
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            return_attention_mask=True,
        )

    def extract_activations(self, sentences, target_indices):
        word_lists = [sent.split() for sent in sentences]

        batch_encoding = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            return_attention_mask=True,
        )

        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        batch_size = input_ids.size(0)
        d_model = hidden_states[0].size(-1)
        activations = torch.empty((batch_size, n_layers, d_model), device=self.device)

        for i in range(batch_size):
            word_id_map = batch_encoding.word_ids(batch_index=i)
            tgt_word_idx = int(target_indices[i])
            positions = [pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx]

            if not positions:
                valid = [pos for pos, wid in enumerate(word_id_map) if wid is not None]
                positions = [valid[-1]] if valid else [0]

            last_pos = positions[-1]

            for layer_idx, layer_states in enumerate(hidden_states):
                activations[i, layer_idx, :] = layer_states[i, last_pos, :]

        return activations.cpu()
