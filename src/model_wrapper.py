import torch
from transformers import AutoModel, AutoTokenizer
from src import config, utils


class ModelWrapper:
    def __init__(self, model_key: str, revision: str = None):
        if model_key not in config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        self.model_config = config.MODEL_CONFIGS[model_key]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Try loading PyTorch weights, else retry from TF
        try:
            self.model = AutoModel.from_pretrained(
                self.model_config["model_name"],
                revision=revision,
                output_hidden_states=True,
                output_attentions=True,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
        except OSError as e:
            msg = str(e).lower()
            if "does not appear to have a file named pytorch_model.bin" in msg and "tensorflow" in msg:
                utils.log_info(f"PyTorch weights not found for '{model_key}', retrying from TensorFlow checkpoint.")
                self.model = AutoModel.from_pretrained(
                    self.model_config["model_name"],
                    revision=revision,
                    from_tf=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    trust_remote_code=self.model_config.get("trust_remote_code", False),
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                raise

        # 2) Tokenizer (fast, else slow fallback)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["tokenizer_name"],
                revision=revision,
                add_prefix_space=True,
                use_fast=True,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
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
                trust_remote_code=self.model_config.get("trust_remote_code", False),
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # 3) Determine where the layers live
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            # BERT/DeBERTa style
            self.layers = self.model.encoder.layer
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "block"):
            # T5/mT5 style
            self.layers = self.model.encoder.block
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Qwen2 or some HF wrapper style
            self.layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            # Pythia, LLaMA, etc.
            self.layers = self.model.layers
        elif hasattr(self.model, "h"):
            # GPT-2 style
            self.layers = self.model.h
        else:
            raise ValueError("Cannot find `.layers` (or `.h`) on the model; unsupported architecture.")

        # Prepare storage for hooks
        n_layers = len(self.layers)
        self.attn_outputs = [None] * n_layers
        self._hook_handles = []

        # 4) Register a forward hook on each layer’s self-attention module
        for idx, layer in enumerate(self.layers):
            # priority: .attention.self → .attention → .self_attn → .attn
            if hasattr(layer, "attention"):
                attn_mod = layer.attention
                # BERT/DeBERTa style
                if hasattr(attn_mod, "self"):
                    target_module = attn_mod.self
                else:
                    # NeoX/Pythia style
                    target_module = attn_mod
            elif hasattr(layer, "layer") and hasattr(layer.layer[0], "SelfAttention"):
                # T5/mT5 style
                target_module = layer.layer[0].SelfAttention
            elif hasattr(layer, "self_attn"):
                # Qwen2, LLaMA, OLMo style
                target_module = layer.self_attn
            elif hasattr(layer, "attn"):
                # GPT-2 style
                target_module = layer.attn
            else:
                raise TypeError(
                    f"Could not find a self-attention submodule in layer {idx} ({layer.__class__.__name__})"
                )

            handle = target_module.register_forward_hook(self._make_hook(idx))
            self._hook_handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs could be a tensor or tuple
            ctx = outputs[0] if isinstance(outputs, tuple) else outputs
            # ctx shape: (batch_size, seq_len, hidden_size)
            self.attn_outputs[layer_idx] = ctx
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
        # Handles mapping from word index to token index. If the underlying model/tokenizer uses word_ids,
        # we take the last token corresponding to the word. Otherwise fall back to the last non-padding token.
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
            positions = [
                pos for pos, wid in enumerate(word_id_map) if wid == tgt_word_idx
            ]
            if not positions:
                valid = [
                    pos for pos, wid in enumerate(word_id_map) if wid is not None
                ]
                positions = [valid[-1]] if valid else [0]
            return positions[-1]
        except (AttributeError, ValueError):
            attention_mask = batch_encoding["attention_mask"]
            non_pad_positions = attention_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            return non_pad_positions[-1].item() if non_pad_positions.numel() > 0 else 0

    def extract_activations(self, sentences, target_indices, use_attention: bool = False):
        """
        Extract hidden-state or attention activations for specified positions/spans in each sentence.

        Args:
            sentences (List[str]): List of sentences.
            target_indices: For each sentence, this can be:
                - An int (the index of the target word);
                - A list of ints (a single span of word indices);
                - A list of lists of ints (multiple spans, e.g. for multi-span tasks). All entries in the batch
                  must have the same number of spans.
            use_attention (bool): If True, extract attention outputs instead of hidden states.

        Returns:
            torch.Tensor: Activations of shape (batch_size, n_layers, hidden_size * n_spans).
        """
        # Normalize sentences into lists of words for word-level alignment
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
        input_ids = batch_encoding["input_ids"].to(self.model.device)
        attention_mask = batch_encoding["attention_mask"].to(self.model.device)

        batch_size = input_ids.size(0)
        n_layers = len(self.layers)
        hidden_size = self.model.config.hidden_size

        # Determine number of spans. We assume homogeneity across the batch.
        # Convert target_indices into a canonical form: list of list of list-of-int positions.
        normalized_targets = []
        n_spans = None
        for idx in target_indices:
            # If element is a list of lists, treat as multi spans.
            if isinstance(idx, list) and idx and all(isinstance(x, list) for x in idx):
                spans = idx
            # If element is a list of ints, treat as a single span.
            elif isinstance(idx, list):
                spans = [idx]
            # Otherwise treat as single index.
            else:
                spans = [[int(idx)]]
            # Update n_spans
            if n_spans is None:
                n_spans = len(spans)
            elif n_spans != len(spans):
                raise ValueError(
                    "Inconsistent number of spans across samples; all examples must have the same number of spans"
                )
            normalized_targets.append(spans)
        if n_spans is None:
            n_spans = 1

        # Prepare storage for activations: shape (batch_size, n_layers, hidden_size * n_spans)
        activations = torch.empty(
            (batch_size, n_layers, hidden_size * n_spans), device=self.model.device
        )

        if use_attention:
            # Clear any stale hooks
            self.attn_outputs = [None] * n_layers

        with torch.no_grad():
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": not use_attention,
                "output_attentions": use_attention,
                "return_dict": True,
            }
            # T5-family models are encoder-decoder and require decoder_input_ids
            if "t5" in self.model_config["model_name"].lower():
                model_kwargs["decoder_input_ids"] = input_ids

            outputs = self.model(**model_kwargs)

        # Hidden states or attention outputs per layer
        if not use_attention:
            hidden_states = outputs.hidden_states

        for i in range(batch_size):
            spans = normalized_targets[i]
            for layer_idx in range(n_layers):
                # Prepare representation vector(s) for this layer and this sample
                span_vectors = []
                if use_attention:
                    ctx = self.attn_outputs[layer_idx]
                    if ctx is None:
                        raise RuntimeError(
                            f"Attention outputs for layer {layer_idx} were not set by hook."
                        )
                for span in spans:
                    # For each span, gather token positions for each word index in span
                    token_positions = []
                    for word_idx in span:
                        # Convert word-level index to token-level position
                        pos = self._get_token_position(i, [word_idx], batch_encoding)
                        token_positions.append(pos)
                    # Compute the representation for this span: mean of token hidden states/attentions
                    if use_attention:
                        # ctx shape: (batch, seq_len, hidden_size)
                        vecs = ctx[i, token_positions]
                    else:
                        vecs = hidden_states[layer_idx][i, token_positions]
                    # Average across positions
                    if vecs.ndim == 1:
                        span_vec = vecs
                    else:
                        span_vec = vecs.mean(dim=0)
                    span_vectors.append(span_vec)
                # Concatenate vectors for all spans
                if len(span_vectors) == 1:
                    concat_vec = span_vectors[0]
                else:
                    concat_vec = torch.cat(span_vectors, dim=-1)
                activations[i, layer_idx] = concat_vec
        return activations.cpu()

    def _extract_byt5_word_position(self, sentence, target_index, batch_encoding, batch_idx):
        words = sentence.split()
        target_word = words[int(target_index)]
        char_start = 0
        for i, word in enumerate(words):
            if i == int(target_index):
                char_end = char_start + len(target_word)
                break
            char_start += len(word) + 1
        tokens = batch_encoding["input_ids"][batch_idx]
        non_pad_positions = (tokens != self.tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
        estimated_pos = min(char_end, len(non_pad_positions) - 1)
        return (
            non_pad_positions[estimated_pos].item()
            if estimated_pos < len(non_pad_positions)
            else non_pad_positions[-1].item()
        )

    def get_layernorm_params(self, layer_idx):
        state_dict = self.model.state_dict()
        weight = None

        ln_key_base = f'layers.{layer_idx}.input_layernorm'
        weight_key = f'model.{ln_key_base}.weight'

        if weight_key in state_dict:
            weight = state_dict[weight_key]

        if weight is None:
            for k in state_dict:
                if k.endswith(f'layers.{layer_idx}.input_layernorm.weight'):
                    weight = state_dict[k]

        if weight is None:
            raise ValueError(f"Could not find LayerNorm weight for layer {layer_idx}")

        return weight