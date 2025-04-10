import torch
from transformers import AutoModel, AutoTokenizer
from src import config

class ModelWrapper:
    def __init__(self, model_key: str):
        if model_key not in config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        self.model_config = config.MODEL_CONFIGS[model_key]
        self.device = torch.device(config.TRAIN_PARAMS["device"] if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_config["model_name"], output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["tokenizer_name"])
        # Fix: If the tokenizer does not have a pad token, set it to the eos_token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, sentences):
        """
        Tokenizes a list of sentences with appropriate padding and truncation.
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
        Given a list of sentences and target token indices (adjusted for the model's tokenization),
        returns a tensor of shape (batch_size, n_layers, d_model) corresponding to the hidden states
        at the target token positions.
        """
        inputs = self.tokenize(sentences)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of layers

        n_layers = len(hidden_states)
        batch_size = input_ids.shape[0]
        d_model = hidden_states[0].shape[-1]
        activations = torch.empty((batch_size, n_layers, d_model), device=self.device)

        # For each example, select the token at the target index (if index is too high, use the last token)
        for i in range(batch_size):
            idx = target_indices[i]
            if idx >= input_ids.shape[1]:
                idx = input_ids.shape[1] - 1
            for l, layer_state in enumerate(hidden_states):
                activations[i, l] = layer_state[i, idx]

        return activations.cpu()
