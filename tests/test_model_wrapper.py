import torch
import numpy as np
import pytest

# We’ll patch the AutoModel/AutoTokenizer imports inside model_wrapper
import src.model_wrapper as mw

# A fake model that returns two layers of constant hidden‐states: ones and twos.
class FakeModel:
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        d_model = 1
        layer0 = torch.ones((batch, seq_len, d_model))
        layer1 = torch.full((batch, seq_len, d_model), 2.0)
        # huggingface returns an object with .hidden_states
        return type("O", (), {"hidden_states": (layer0, layer1)})

# A fake tokenizer that treats each word as one token, pads to length=3,
# exposes pad_token/eos_token, and implements __getitem__+word_ids()
class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, word_lists=None, is_split_into_words=None,
                 return_tensors=None, padding=None, truncation=None,
                 max_length=None, return_attention_mask=None, **kwargs):
        # assume word_lists is a list of lists of strings
        wl = word_lists
        batch = len(wl)
        seq_len = 3
        # dummy tensors
        input_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch, seq_len), dtype=torch.long)

        class BE:
            def __init__(self, input_ids, attention_mask, wl):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self._wl = wl
            def __getitem__(self, key):
                return getattr(self, key)
            def word_ids(self, batch_index):
                seq_len = self.input_ids.shape[1]
                return [
                    idx if idx < len(self._wl[batch_index]) else None
                    for idx in range(seq_len)
                ]

        return BE(input_ids, attention_mask, wl)

@pytest.fixture(autouse=True)
def patch_auto(monkeypatch):
    # patch the two names in model_wrapper module
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()
    monkeypatch.setattr(mw, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(mw, "AutoTokenizer", FakeAutoTokenizer)

def test_extract_activations_basic():
    wrapper = mw.ModelWrapper("gpt2")
    sents = ["a b b", "c"]
    # for first sent target index=1 → picks only token position 1
    # for second       target index=0 → picks only token position 0
    acts = wrapper.extract_activations(sents, [1, 0])
    # should be (batch=2, layers=2, d_model=1)
    assert isinstance(acts, torch.Tensor)
    assert acts.shape == (2, 2, 1)
    # layer0 should be all ones, layer1 all twos
    expected = torch.tensor([[[1.],[2.]], [[1.],[2.]]])
    assert torch.allclose(acts, expected)

def test_extract_activations_fallback():
    wrapper = mw.ModelWrapper("gpt2")
    # give a target index out of range → no word_ids match → fallback to last non‐None
    acts = wrapper.extract_activations(["x y"], [10])
    # shape (1,2,1)
    assert acts.shape == (1,2,1)
    # still ones / twos
    assert torch.allclose(acts, torch.tensor([[[1.],[2.]]]))
