import sys
import os
import importlib
from transformers import AutoConfig

sys.path.append(os.path.dirname(__file__))
config = importlib.import_module("config")

def get_num_layers(model_config):
    try:
        cfg = AutoConfig.from_pretrained(model_config["model_name"])
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "num_layers_encoder"]:
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        if hasattr(cfg, "architectures"):
            return cfg.architectures
        return "Unknown"
    except Exception as e:
        return f"Error: {e}"

results = []
for key, model_cfg in config.MODEL_CONFIGS.items():
    num_layers = get_num_layers(model_cfg)
    results.append((key, num_layers))

print(r"\begin{table}")
print(r"  \centering")
print(r"  \begin{tabular}{lr}")
print(r"    \hline")
print(r"    \textbf{Model} & \textbf{\# of Layers} \\")
print(r"    \hline")
for model, layers in results:
    print(f"    {model} & {layers} \\\\")
print(r"    \hline")
print(r"  \end{tabular}")
print(r"  \caption{Number of layers for each model}")
print(r"  \label{tab:model_num_layers}")
print(r"\end{table}")
