import os
import pytest
import src.config as config

def test_model_configs_keys():
    # These are the two you ran: 
    assert "pythia1.4b" in config.MODEL_CONFIGS
    assert "qwen2"       in config.MODEL_CONFIGS

@pytest.mark.parametrize("model_key", config.MODEL_CONFIGS)
def test_each_model_has_necessary_fields(model_key):
    cfg = config.MODEL_CONFIGS[model_key]
    # every config must specify these
    for field in ("model_name", "tokenizer_name", "max_length", "batch_size"):
        assert field in cfg

def test_output_dir_exists_and_writable(tmp_path, monkeypatch):
    # Override BASE_DIR so OUTPUT_DIR is inside tmp_path
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    # re-import to pick up env override
    import importlib
    import src.config
    importlib.reload(src.config)
    assert os.path.isdir(src.config.OUTPUT_DIR)
    # can write a dummy file
    p = os.path.join(src.config.OUTPUT_DIR, "foo.txt")
    with open(p, "w") as f:
        f.write("ok")
    with open(p) as f:
        assert f.read() == "ok"
