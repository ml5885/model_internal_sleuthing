# src/model_wrapper.py

import torch
from transformers import AutoModel, AutoTokenizer
from src import config

class ModelWrapper:
    def __init__(self, model_key: str):
        if model_key not in config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        self.model_config = config.MODEL_CONFIGS[model_key]
        # Pick GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_config["model_name"],
            output_hidden_states=True
        )

        # Load tokenizer; GPT2TokenizerFast needs add_prefix_space when pre‐tokenizing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["tokenizer_name"],
            add_prefix_space=True
        )
        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, sentences):
        """
        Tokenizes a list of raw sentences (strings) normally, for things like batching
        or attention masks.  Does NOT do word‐level splitting.
        """
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            return_attention_mask=True,
        )

    def extract_activations(self, sentences, target_indices):
        """
        Given a list of sentences (strings) and a parallel list of target word‐indices
        (i.e. which word in sentence.split() is the target), returns a tensor of shape
        (batch_size, n_layers, d_model) where for each example and each layer we
        average the hidden states over all tokens that correspond to that target word.
        """
        # First break into words so tokenizer can tell us word_ids
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

        # Run model
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple: (batch, seq_len, d_model) for each layer

        n_layers = len(hidden_states)
        batch_size = input_ids.shape[0]
        d_model = hidden_states[0].size(-1)

        # Prepare output tensor
        activations = torch.empty((batch_size, n_layers, d_model), device=self.device)

        # For each example in batch, gather all token‑positions mapping to the target word
        for i in range(batch_size):
            word_id_map = batch_encoding.word_ids(batch_index=i)  # list of length seq_len, entries in {None, 0,1,2,...}
            tgt_word_idx = int(target_indices[i])

            # All token positions whose word_id == tgt_word_idx
            positions = [pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx]

            # Fallback: if nothing matched, pick the last real token
            if not positions:
                valid = [pos for pos, wid in enumerate(word_id_map) if wid is not None]
                positions = [valid[-1]] if valid else [0]

            # For each layer, average over those positions
            for layer_idx, layer_states in enumerate(hidden_states):
                # layer_states is (batch, seq_len, d_model)
                token_states = layer_states[i, positions, :]      # (n_pos, d_model)
                activations[i, layer_idx, :] = token_states.mean(dim=0)

        return activations.cpu()
