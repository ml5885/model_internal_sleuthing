import torch
from transformers import AutoModel, AutoTokenizer
from src import config, utils
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import json

class ModelWrapper:
    def __init__(self, model_key: str, revision: str = None):
        if model_key not in config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        self.model_config = config.MODEL_CONFIGS[model_key]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(
            self.model_config["model_name"],
            revision=revision,
            output_hidden_states=True,
            output_attentions=True,
            trust_remote_code=self.model_config.get("trust_remote_code", False)
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["tokenizer_name"],
                revision=revision,
                add_prefix_space=True,
                use_fast=True,
                trust_remote_code=self.model_config.get("trust_remote_code", False)
            )
        except TypeError:
            # Fallback for models like CohereForAI/aya-101
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["tokenizer_name"],
                revision=revision,
                add_prefix_space=True,
                use_fast=False,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
                legacy=True
            )
        except Exception as fast_err:
            utils.log_info(
                f"Fast tokenizer load failed for '{model_key}' ({fast_err}); falling back to slow Python tokenizer."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["tokenizer_name"],
                revision=revision,
                add_prefix_space=True,
                use_fast=False,
                trust_remote_code=self.model_config.get("trust_remote_code", False)
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

        # Determine where the layer list lives (encoder vs. decoder)
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            self.layers = self.model.encoder.layer
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            self.layers = self.model.layers
        else:
            raise ValueError("Cannot find `.layers` on the model; unsupported architecture.")

        # Prepare storage for n_layers hooks
        n_layers = len(self.layers)
        self.attn_outputs = [None] * n_layers
        self._hook_handles = []

        # Register a forward hook on each layer's self-attention module
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                # encoder-style: layer.attention.self
                target_module = layer.attention.self
            elif hasattr(layer, "self_attn"):
                # decoder-style (e.g. Qwen2DecoderLayer): layer.self_attn
                target_module = layer.self_attn
            else:
                raise TypeError(
                    f"Could not find a self-attention submodule in layer {idx} ({layer.__class__.__name__})"
                )

            handle = target_module.register_forward_hook(self._make_hook(idx))
            self._hook_handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            context = outputs[0] if isinstance(outputs, tuple) else outputs
            # context shape: (batch_size, seq_len, hidden_size)
            self.attn_outputs[layer_idx] = context
        return hook

    def tokenize(self, sentences):
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            return_attention_mask=True,
        )

    def _get_token_position(self, batch_idx, target_indices, batch_encoding):
        # Special handling for byte-level models like ByT5
        if 'byt5' in self.model_config["model_name"].lower():
            return self._extract_byt5_word_position(
                batch_encoding.encodings[batch_idx].text,
                target_indices[batch_idx],
                batch_encoding,
                batch_idx
            )
        try:
            word_id_map = batch_encoding.word_ids(batch_index=batch_idx)
            tgt_word_idx = int(target_indices[batch_idx])
            positions = [pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx]
            if not positions:
                valid = [pos for pos, wid in enumerate(word_id_map) if wid is not None]
                positions = [valid[-1]] if valid else [0]
            return positions[-1]
        except (AttributeError, ValueError):
            # fallback: last non-pad token
            attention_mask = batch_encoding["attention_mask"]
            non_pad_positions = attention_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            return non_pad_positions[-1].item() if non_pad_positions.numel() > 0 else 0

    def extract_activations(self, sentences, target_indices, use_attention=False):
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

        batch_size = input_ids.size(0)
        n_layers = len(self.layers)
        hidden_size = self.model.config.hidden_size

        if use_attention:
            self.attn_outputs = [None] * n_layers
            activations = torch.empty(
                (batch_size, n_layers, hidden_size), device=self.device
            )
        else:
            activations = torch.empty(
                (batch_size, n_layers, hidden_size), device=self.device
            )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=not use_attention,
                output_attentions=False,
                return_dict=True
            )

        if not use_attention:
            hidden_states = outputs.hidden_states

        for i in range(batch_size):
            last_pos = self._get_token_position(i, target_indices, batch_encoding)
            for layer_idx in range(n_layers):
                if use_attention:
                    ctx = self.attn_outputs[layer_idx]
                    if ctx is None:
                        raise RuntimeError(
                            f"Attention outputs for layer {layer_idx} were not set by hook."
                        )
                    activations[i, layer_idx] = ctx[i, last_pos]
                else:
                    activations[i, layer_idx] = hidden_states[layer_idx][i, last_pos]

        return activations.cpu()

    def _extract_byt5_word_position(self, sentence, target_index, batch_encoding, batch_idx):
        """Extract the token position for the target word in ByT5 byte-level tokenization."""
        words = sentence.split()
        target_word = words[int(target_index)]
        
        # Find character span of target word in original sentence
        char_start = 0
        for i, word in enumerate(words):
            if i == int(target_index):
                char_end = char_start + len(target_word)
                break
            char_start += len(word) + 1  # +1 for space
        
        # Map character positions to token positions
        # This is approximate - ByT5 encoding can be complex
        tokens = batch_encoding["input_ids"][batch_idx]
        non_pad_positions = (tokens != self.tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
        
        # Use a heuristic: take the token position that roughly corresponds to the end of target word
        # This is simplified - a more robust approach would decode tokens to find exact boundaries
        estimated_pos = min(char_end, len(non_pad_positions) - 1)
        
        return non_pad_positions[estimated_pos].item() if estimated_pos < len(non_pad_positions) else non_pad_positions[-1].item()

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
                if k.endswith(f'layers.{layer_idx+1}.input_layernorm.bias'):
                    bias = state_dict[k]
        if weight is None or bias is None:
            raise ValueError(f"Could not find LayerNorm params for layer {layer_idx+1}")
        return weight, bias
        if weight is None or bias is None:
            raise ValueError(f"Could not find LayerNorm params for layer {layer_idx+1}")
        return weight, bias
