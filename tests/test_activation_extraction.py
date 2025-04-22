import os
import numpy as np
import pandas as pd
import torch
import pytest
from src.activation_extraction import extract_and_save

# Dummy wrapper that returns a tensor where each example i is filled with float(i)
class DummyWrapper:
    def __init__(self, key): pass
    def extract_activations(self, sents, idxs):
        batch = len(sents)
        # 2 layers, d_model=3
        t = torch.zeros((batch, 2, 3))
        for i in range(batch):
            t[i] = float(i)
        return t

@pytest.fixture(autouse=True)
def patch_wrapper_and_batch(monkeypatch):
    # patch ModelWrapper
    import src.activation_extraction as ae
    monkeypatch.setattr(ae, "ModelWrapper", DummyWrapper)
    # shrink batch size so we get multiple shards
    import src.config as cfg
    cfg.MODEL_CONFIGS["gpt2"]["batch_size"] = 2

def test_extract_and_save_sharding(tmp_path):
    # build a 5‐row CSV
    df = pd.DataFrame({
        "Sentence": ["s1","s2","s3","s4","s5"],
        "Target Index": [0,0,0,0,0]
    })
    data_csv = tmp_path / "in.csv"
    df.to_csv(data_csv, index=False)

    outdir = tmp_path / "out"
    extract_and_save(str(data_csv), str(outdir), "gpt2")

    files = sorted(os.listdir(outdir))
    # 5 examples, batch_size=2 → ceil(5/2)=3 shards
    assert len(files) == 3

    for shard_idx, fname in enumerate(files):
        arr = np.load(os.path.join(outdir, fname))["activations"]
        # first two shards have 2 rows, last has 1
        expected_rows = 2 if shard_idx < 2 else 1
        assert arr.shape == (expected_rows, 2, 3)
        # DummyWrapper numbers each example *within* the shard as 0,1,...
        for local_i in range(expected_rows):
            assert np.all(arr[local_i] == float(local_i))
