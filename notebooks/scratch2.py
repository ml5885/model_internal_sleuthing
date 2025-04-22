import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----- HELPERS -----
def load_results(dataset, model, task_name):
    variants = [task_name]
    tn = task_name.lower()
    if "inflect" in tn:
        variants += [v for v in ("multiclass_inflection", "inflection") if v not in variants]
    for v in variants:
        for suffix in ("", "_v1"):
            path = f"../output/probes/{dataset}_{model}_{v}_reg{suffix}"
            if os.path.exists(path):
                csvs = glob.glob(os.path.join(path, "*_results.csv"))
                if csvs:
                    return pd.read_csv(csvs[0])
    return None

def extract_data(df, task_name):
    if df is None:
        return None
    layers = df["Layer"].astype(int).values
    cols = {c.lower(): c for c in df.columns}
    tn = task_name.lower()
    if "lexeme" in tn:
        tcol = cols.get("lexeme_accuracy", cols.get("lexeme_task"))
    else:
        tcol = cols.get("inflection_accuracy", cols.get("inflection_task"))
    acc = df[tcol].values * 100
    order = np.argsort(layers)
    return acc[order]

# ----- MAIN SCRIPT -----
dataset    = "ud_gum_dataset"
tasks      = ["lexeme", "multiclass_inflection"]
probe_dir  = "../output/probes/"

# 1. Discover all models
pattern = re.compile(rf"{dataset}_(.+?)_(?:lexeme|multiclass_inflection|inflection)_reg")
models = sorted({
    m.group(1)
    for entry in os.listdir(probe_dir)
    if (m := pattern.match(entry))
})

# 2. Load results for each model & task
results = {}
for m in models:
    accs = {}
    for t in tasks:
        df = load_results(dataset, m, t)
        acc = extract_data(df, t)
        if acc is not None:
            accs[t] = acc
    if set(accs) == set(tasks):
        results[m] = accs

# 3. Compute average accuracies
names   = []
lex_avg = []
inf_avg = []
for m, accs in results.items():
    names.append(m)
    lex_avg.append(accs["lexeme"].mean())
    inf_avg.append(accs["multiclass_inflection"].mean())

# 4. Define vocab sizes
vocab_sizes = {
    "bert-base-uncased":   30522,
    "bert-large-uncased":  30000,
    "gpt2":                50257,
    "pythia1.4b":          50304,
    "qwen2":             151936,
    "gemma2b":           256000
}

# Colors for each model
model_colors = {
    "gpt2":               "tab:blue",
    "pythia1.4b":         "tab:orange",
    "gemma2b":            "tab:green",
    "qwen2":              "tab:purple",
    "bert-base-uncased":  "tab:red",
    "bert-large-uncased": "tab:olive"
}

# 5. Filter to models present in both results & vocab_sizes
filtered = [
    (m, la, ia, vocab_sizes[m])
    for m, la, ia in zip(names, lex_avg, inf_avg)
    if m in vocab_sizes
]
if not filtered:
    raise ValueError("No overlapping models found between results and vocab_sizes.")

# 6. Unpack filtered data
names_filt, lex_avg_filt, inf_avg_filt, sizes = zip(*filtered)

# 8. Plot
fig = plt.figure(figsize=(12, 6))  # Create figure object explicitly
main_ax = fig.add_axes([0.1, 0.1, 1.0, 1.0])  # [left, bottom, width, height]

for m, la, ia, sz in zip(names_filt, lex_avg_filt, inf_avg_filt, sizes):
    color = model_colors[m]
    main_ax.scatter(sz, la,
                   marker="o", s=100,
                   color=color, edgecolor="k")
    main_ax.scatter(sz, ia,
                   marker="s", s=100,
                   color=color, edgecolor="k")
main_ax.set_xscale("log")
main_ax.set_xlabel("Vocabulary Size", fontsize=16)
main_ax.set_ylabel("Average Accuracy (%)", fontsize=16)
main_ax.set_title("Model Vocabulary Size vs. Average Probing Accuracy", fontsize=18)
main_ax.grid(True, linestyle="--", alpha=0.6)

# 9. Legends
# a) Model colors
model_handles = [
    Line2D([0], [0],
           marker="o",
           color=col,
           linestyle="",
           markersize=10,
           markeredgecolor="k",
           label=m)
    for m, col in model_colors.items()
]
legend1 = main_ax.legend(handles=model_handles,
                        title="Model",
                        loc="center left",
                        bbox_to_anchor=(1, 0.6))
main_ax.add_artist(legend1)

# b) Task shapes
task_handles = [
    Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=10, markeredgecolor="k", label="Lexeme"),
    Line2D([0], [0], marker="s", color="gray", linestyle="", markersize=10, markeredgecolor="k", label="Inflection")
]
main_ax.legend(handles=task_handles,
              title="Task",
              loc="center left",
              bbox_to_anchor=(1.05, 0.3))

plt.savefig("vocab_size_vs_avg_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

print("Done: saved vocab_size_vs_avg_accuracy.png")
