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

        self.model = AutoModel.from_pretrained(
            self.model_config["model_name"],
            output_hidden_states=True,
            use_auth_token=True,
            trust_remote_code=self.model_config.get("trust_remote_code", False) # for deepseek
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["tokenizer_name"],
                add_prefix_space=True,
                use_fast=True,
                use_auth_token=True,
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
                    use_auth_token=True,
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
            except Exception as slow_err:
                utils.log_info(f"Slow tokenizer load failed for '{model_key}' ({slow_err}).")
                raise ImportError("Failed to load tokenizer.") from slow_err

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
            word_id_map = batch_encoding.word_ids(batch_index=i) # maps token ids to word ids
            tgt_word_idx = int(target_indices[i])
            positions = [pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx]

            # handle if word was removed somehow
            if not positions:
                valid = [pos for pos, wid in enumerate(word_id_map) if wid is not None]
                positions = [valid[-1]] if valid else [0]

            last_pos = positions[-1]

            for layer_idx, layer_states in enumerate(hidden_states):
                activations[i, layer_idx, :] = layer_states[i, last_pos, :]
                # activations[i, layer_idx, :] = layer_states[i, positions, :].mean(dim=0)
                
        return activations.cpu()

    def get_layernorm_params(self, layer_idx):
        ln_name = f'model.layers.{layer_idx+1}.input_layernorm'
        state_dict = self.model.state_dict()
        weight, bias = None, None
        for k in state_dict:
            if k.endswith(f'layers.{layer_idx+1}.input_layernorm.weight'):
                weight = state_dict[k]
            if k.endswith(f'layers.{layer_idx+1}.input_layernorm.bias'):
                bias = state_dict[k]
        if weight is None or bias is None:
            for k in state_dict:
                if k.endswith(f'layers.{layer_idx+1}.input_layernorm.weight'):
                    weight = state_dict[k]
                if k.endswith(f'layers.{layer_idx+1}.input_layernorm.bias'):
                    bias = state_dict[k]
        if weight is None or bias is None:
            raise ValueError(f"Could not find LayerNorm params for layer {layer_idx+1}")
        return weight, bias
