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

    def tokenize(self, sentences):
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            return_attention_mask=True,
        )

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

        with torch.no_grad():
            if hasattr(self.model, 'encoder'):
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                if use_attention:
                    attentions = encoder_outputs.attentions
                else:
                    hidden_states = encoder_outputs.hidden_states
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                if use_attention:
                    attentions = outputs.attentions
                else:
                    hidden_states = outputs.hidden_states

        batch_size = input_ids.size(0)
        max_length = self.model_config["max_length"]

        if use_attention:
            n_layers = len(attentions)
            n_heads = attentions[0].size(1)
            d_attn = n_heads * max_length  # Always use fixed max_length
            activations = torch.empty((batch_size, n_layers, d_attn), device=self.device)
        else:
            n_layers = len(hidden_states)
            d_model = hidden_states[0].size(-1)
            activations = torch.empty((batch_size, n_layers, d_model), device=self.device)

        for i in range(batch_size):
            # Special handling for byte-level models like ByT5
            if 'byt5' in self.model_config["model_name"].lower():
                last_pos = self._extract_byt5_word_position(sentences[i], target_indices[i], batch_encoding, i)
            else:
                # try to map tokens back to word indices (fast tokenizer)
                try:
                    word_id_map = batch_encoding.word_ids(batch_index=i)
                    tgt_word_idx = int(target_indices[i])
                    positions = [pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx]

                    if not positions:
                        valid = [pos for pos, wid in enumerate(word_id_map) if wid is not None]
                        positions = [valid[-1]] if valid else [0]

                    last_pos = positions[-1]
                except (AttributeError, ValueError):
                    # slow tokenizer (e.g. ByT5) doesn't support word_ids
                    # just use last non-pad token
                    non_pad_positions = attention_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    last_pos = non_pad_positions[-1].item() if non_pad_positions.numel() > 0 else 0

            for layer_idx in range(n_layers):
                if use_attention:
                    # A: (batch, heads, seq, seq)
                    A = attentions[layer_idx]
                    # Extract the full attention vector from last_pos to all tokens, for all heads
                    # shape: (n_heads, seq_len)
                    v = A[i, :, last_pos, :]  # (n_heads, seq_len)
                    seq_len = v.size(-1)
                    if seq_len < max_length:
                        # pad with zeros at the end
                        pad = torch.zeros((n_heads, max_length - seq_len), device=v.device, dtype=v.dtype)
                        v_padded = torch.cat([v, pad], dim=-1)
                    else:
                        v_padded = v[:, :max_length]
                    activations[i, layer_idx, :] = v_padded.flatten()  # (n_heads * max_length,)
                else:
                    layer_states = hidden_states[layer_idx]
                    activations[i, layer_idx, :] = layer_states[i, last_pos, :]
                
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
