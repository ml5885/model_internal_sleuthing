import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re

import datetime as dt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

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
    print(f"No results found for {dataset}_{model}_{task_name}")
    return None

def extract_data(df, task_name):
    if df is None:
        return None, None, None
    layers = df["Layer"].astype(int).values
    cols = {c.lower(): c for c in df.columns}
    tn = task_name.lower()
    if "lexeme" in tn:
        if "lexeme_accuracy" in cols:
            task_col    = cols["lexeme_accuracy"]
            control_col = cols["lexeme_controlaccuracy"]
        elif "lexeme_task" in cols and "lexeme_control" in cols:
            task_col    = cols["lexeme_task"]
            control_col = cols["lexeme_control"]
        else:
            raise ValueError(f"No lexeme columns in {list(df.columns)}")
    elif "inflect" in tn:
        if "inflection_accuracy" in cols:
            task_col    = cols["inflection_accuracy"]
            control_col = cols["inflection_controlaccuracy"]
        elif "inflection_task" in cols and "inflection_control" in cols:
            task_col    = cols["inflection_task"]
            control_col = cols["inflection_control"]
        else:
            raise ValueError(f"No inflection columns in {list(df.columns)}")
    else:
        raise ValueError(f"Unknown task {task_name!r}")
    task_acc    = df[task_col].values * 100
    control_acc = df[control_col].values * 100
    return layers, task_acc, control_acc

dataset = "ud_gum_dataset"
tasks   = ["lexeme", "multiclass_inflection"]

probe_dir = "../output/probes/"
model_pattern = re.compile(rf"{dataset}_(.+?)_(lexeme|multiclass_inflection|inflection)_reg")

model_names = set()
for entry in os.listdir(probe_dir):
    m = model_pattern.match(entry)
    if m:
        model_names.add(m.group(1))
models = sorted(model_names)

results = {m: {} for m in models}

for m in models:
    for t in tasks:
        df_res = load_results(dataset, m, t)
        layers, task_acc, _ = extract_data(df_res, t)
        if layers is not None:
            idx = np.argsort(layers)
            results[m][t] = {
                "layers":   layers[idx],
                "task_acc": task_acc[idx]
            }

for m in models:
    if "lexeme" in results[m] and "multiclass_inflection" in results[m]:
        lex = results[m]["lexeme"]
        inf = results[m]["multiclass_inflection"]
        plt.figure(figsize=(10, 7))
        plt.plot(
            lex["layers"],
            lex["task_acc"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Lexeme Accuracy",
            color="tab:blue"
        )
        plt.plot(
            inf["layers"],
            inf["task_acc"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Inflection Accuracy",
            color="tab:orange"
        )
        plt.xlabel("Layer", fontsize=22)
        plt.ylabel("Accuracy (%)", fontsize=22)
        plt.ylim(0, 100)
        plt.title(f"{m} Probing Accuracy", fontsize=26)
        plt.legend(fontsize=18)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{m}_lex_vs_inf_line.png", bbox_inches="tight")
        plt.close()

print("Done: generated lexeme vs inflection line plots for all models.")

valid_models = [
    m for m in results
    if "lexeme" in results[m] and "multiclass_inflection" in results[m]
]

all_layers = sorted(set(
    np.concatenate([results[m]["lexeme"]["layers"] for m in valid_models])
))

diff_matrix = np.full((len(valid_models), len(all_layers)), np.nan)
for i, m in enumerate(valid_models):
    lex_data = results[m]["lexeme"]
    inf_data = results[m]["multiclass_inflection"]
    for j, layer in enumerate(all_layers):
        if layer in lex_data["layers"] and layer in inf_data["layers"]:
            lv = lex_data["task_acc"][np.where(lex_data["layers"] == layer)[0][0]]
            iv = inf_data["task_acc"][np.where(inf_data["layers"] == layer)[0][0]]
            diff_matrix[i, j] = lv - iv

delta_df = pd.DataFrame(
    diff_matrix,
    index=valid_models,
    columns=all_layers
)

plt.figure(figsize=(12, 6))
plt.imshow(delta_df, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(abs(diff_matrix)), vmax=np.nanmax(abs(diff_matrix)))
plt.colorbar(label="Lexeme – Inflection Acc (%)")
plt.xticks(ticks=np.arange(len(all_layers)), labels=all_layers, rotation=45)
plt.yticks(ticks=np.arange(len(valid_models)), labels=valid_models)
plt.xlabel("Layer")
plt.ylabel("Model")
plt.title("Heatmap of Lexeme vs Inflection Accuracy Difference")
for i, m in enumerate(valid_models):
    for j, layer in enumerate(all_layers):
        val = delta_df.iat[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(val) > 5 else "black")
plt.tight_layout()
plt.savefig("lexeme_inflection_diff_heatmap.png")
plt.close()

print("Done: saved line plots and heatmap for all models.")

df = pd.read_csv('../data/ud_gum_dataset.csv')
lemma_count = df['Lemma'].nunique()
inflection_count = df['Inflection Label'].nunique()
inflection_dist = df['Inflection Label'].value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=inflection_dist.index, y=inflection_dist.values)
plt.title('Distribution of Inflection Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('inflection_dist.png')
category_dist = df['Category'].value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=category_dist.index, y=category_dist.values)
plt.title('Distribution of Word Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_dist.png')
print(f"Dataset contains {lemma_count} unique lemmas across {inflection_count} inflection types")

release_dates = {
    "bert-base-uncased":  "2018-10-31",
    "bert-large-uncased": "2018-10-31",
    "gpt2":               "2019-02-14",
    "pythia1.4b":         "2023-05-31",
    "qwen2":              "2024-06-06",
    "gemma2b":            "2024-06-27"
}

model_colors = {
    "gpt2":               "tab:blue",
    "pythia1.4b":         "tab:orange",
    "gemma2b":            "tab:green",
    "qwen2":              "tab:purple",
    "bert-base-uncased":  "tab:red",
    "bert-large-uncased": "tab:olive"
}

names, dates, lex_avg, inf_avg = [], [], [], []
for name, date_str in release_dates.items():
    if name in results and "lexeme" in results[name] and "multiclass_inflection" in results[name]:
        names.append(name)
        dates.append(dt.datetime.fromisoformat(date_str))
        lex_vals = results[name]["lexeme"]["task_acc"]
        inf_vals = results[name]["multiclass_inflection"]["task_acc"]
        lex_avg.append(np.mean(lex_vals))
        inf_avg.append(np.mean(inf_vals))

fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(right=0.75)
for name, date, la, ia in zip(names, dates, lex_avg, inf_avg):
    ax.scatter(date, la,
               marker="o", s=100,
               color=model_colors[name], edgecolor="k",
               label=name if name not in ax.get_legend_handles_labels()[1] else "")
    ax.scatter(date, ia,
               marker="s", s=100,
               color=model_colors[name], edgecolor="k")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate()
ax.set_ylim(0, 100)
ax.set_xlabel("Release Date")
ax.set_ylabel("Average Accuracy (%)")
ax.set_title("Model Release Date vs. Average Probing Accuracy")
ax.grid(True, linestyle="--", alpha=0.6)
model_handles = [
    Line2D([0], [0],
           marker="o",
           color=model_colors[m],
           linestyle="",
           markersize=10,
           markeredgecolor="k",
           label=m)
    for m in names
]
legend1 = ax.legend(
    handles=model_handles,
    title="Model",
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)
ax.add_artist(legend1)
task_handles = [
    Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=10, label="Lexeme"),
    Line2D([0], [0], marker="s", color="gray", linestyle="", markersize=10, label="Inflection")
]
ax.legend(
    handles=task_handles,
    title="Task",
    loc="lower left",
    bbox_to_anchor=(1.02, 0)
)
plt.tight_layout()
plt.savefig("release_date_vs_avg_accuracy_scatter.png", bbox_inches="tight")
plt.close()

print("Done: saved wide scatter (model=colour, task=shape) of release date vs. avg accuracy.")

from tqdm import tqdm

def list_shards(path_or_dir):
    files = [f for f in os.listdir(path_or_dir)
             if f.endswith('.npz') and 'activations_part' in f]
    def idx(fname):
        m = re.search(r'part_?(\d+)', fname)
        return int(m.group(1)) if m else -1
    files.sort(key=idx)
    return [os.path.join(path_or_dir, f) for f in files]

def shard_loader(shard_path, layer_idx):
    arr = np.load(shard_path, mmap_mode='r')
    X = arr['activations']
    return X[:, layer_idx, :]

def avg_pairwise_cosine_stream(normed_shard_list, labels):
    uniq = np.unique(labels)
    d_model = normed_shard_list[0].shape[1]
    sums = np.zeros((uniq.size, d_model), dtype=np.float64)
    counts = np.zeros(uniq.size, dtype=np.int64)
    offset = 0
    for Xn in tqdm(normed_shard_list, desc="  computing avg cosine", leave=False):
        B = Xn.shape[0]
        slice_labels = labels[offset:offset+B]
        offset += B
        for lbl in uniq:
            mask = (slice_labels == lbl)
            if not mask.any():
                continue
            sums[lbl]  += Xn[mask].sum(axis=0)
            counts[lbl] += int(mask.sum())
    numer = ((np.linalg.norm(sums, axis=1)**2 - counts) / 2).sum()
    denom = ((counts * (counts - 1)) / 2).sum()
    return float(numer / denom) if denom > 0 else 0.0

labels_df = pd.read_csv('../data/ud_gum_dataset.csv')
inf_labels = pd.Categorical(labels_df["Inflection Label"]).codes
lex_labels = pd.Categorical(labels_df["Lemma"]).codes

models = ['gpt2', 'qwen2', 'gemma2b', 'bert-base-uncased', 'pythia1.4b']
for m in models:
    act_dir = f"../output/{m}_{dataset}_reps"
    shards = list_shards(act_dir)
    if not shards:
        print(f"Skipping {m}: no shards in {act_dir}")
        continue
    sample = np.load(shards[0], mmap_mode='r')['activations']
    n_layers = sample.shape[1]
    cos_inf = []
    cos_lex = []
    for layer in tqdm(range(n_layers), desc=f"Processing {m}", leave=False):
        normed = []
        for sp in shards:
            X = shard_loader(sp, layer)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            normed.append(Xn)
        cos_inf.append(avg_pairwise_cosine_stream(normed, inf_labels))
        cos_lex.append(avg_pairwise_cosine_stream(normed, lex_labels))
    layers = np.arange(n_layers)
    plt.figure(figsize=(8, 5))
    plt.plot(layers, cos_lex,  'o-', label="Lexeme Cosine",    color="tab:blue")
    plt.plot(layers, cos_inf,  's--', label="Inflection Cosine", color="tab:orange")
    plt.xlabel("Layer")
    plt.ylabel("Avg Pairwise Cosine")
    plt.ylim(0, 1)
    plt.title(f"{m}: Average Pairwise Cosine by Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{m}_cosine_similarity.png")
    plt.close()

print("Done: saved cosine similarity plots for all models.")
