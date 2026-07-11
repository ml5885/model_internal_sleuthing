"""
Parse the multilingual scalar-mix results (output/probes_multiling) into a tidy
table of (model, language, task, cog, rel_cog, n_layers, acc, selectivity), then
render cross-model COG heatmaps (relative depth) per language.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"]

MODELS = ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large", "gpt2-xl",
          "gpt2-large", "gpt2", "pythia-6.9b-tulu", "pythia-6.9b", "olmo2-7b-instruct",
          "olmo2-7b", "gemma2b-it", "gemma2b", "qwen2-instruct", "qwen2.5-7B-instruct",
          "qwen2.5-7B", "qwen2", "llama3-8b-instruct", "llama3-8b", "mt5",
          "goldfish_eng_latn_1000mb", "goldfish_zho_hans_1000mb", "goldfish_deu_latn_1000mb",
          "goldfish_fra_latn_1000mb", "goldfish_rus_cyrl_1000mb", "goldfish_tur_latn_1000mb"]
TASKS = ["pos", "dep", "ner", "constituents", "coref", "srl", "relation"]

# dataset prefix -> language
DS_LANG = {"ud_gum": "en", "up_ewt": "en", "semeval2010": "en",
           "ud_chinese": "zh", "msra": "zh", "up_chinese": "zh", "duie": "zh",
           "ud_turkish": "tr", "turkish_wikiner": "tr", "corefud_turkish": "tr",
           "ud_french": "fr", "wikiner_fr": "fr", "up_french": "fr", "corefud_french": "fr", "redfm_fr": "fr",
           "ud_russian": "ru", "nerel": "ru", "rucoco": "ru",
           "ud_german": "de", "germeval": "de", "up_german": "de", "corefud_german": "de", "redfm_de": "de"}


def lang_of(dataset_id):
    for pref, lg in DS_LANG.items():
        if dataset_id.startswith(pref):
            return lg
    return "?"


def parse():
    rows = []
    for d in glob.glob("output/probes_multiling/*_scalarmix"):
        base = os.path.basename(d)[:-len("_scalarmix")]
        # base = {dataset_id}_{model}_{task}; task is the last token
        task = base.rsplit("_", 1)[1]
        if task not in TASKS:
            continue
        rest = base[:-(len(task) + 1)]                       # {dataset_id}_{model}
        model = next((m for m in sorted(MODELS, key=len, reverse=True)
                      if rest.endswith("_" + m)), None)
        if model is None:
            continue
        dataset_id = rest[:-(len(model) + 1)]
        wf = os.path.join(d, "scalarmix_weights.json")
        if not os.path.exists(wf):
            continue
        w = json.load(open(wf))
        acc = sel = np.nan
        pr = os.path.join(d, "probe_results.npz")
        if os.path.exists(pr):
            r = np.load(pr, allow_pickle=True)["results"].item().get("scalarmix", {})
            acc = r.get(f"{task}_acc", np.nan)
            sel = r.get(f"{task}_selectivity", np.nan)
        rows.append({"model": model, "lang": lang_of(dataset_id), "task": task,
                     "cog": w["cog"], "n_layers": w["n_layers"],
                     "rel_cog": w["cog"] / (w["n_layers"] - 1), "acc": acc, "selectivity": sel})
    return pd.DataFrame(rows)


def heatmap(df, lang, out):
    sub = df[df.lang == lang]
    if sub.empty:
        return
    piv = sub.pivot_table(index="task", columns="model", values="rel_cog")
    tasks = [t for t in TASKS if t in piv.index]
    models = [m for m in MODELS if m in piv.columns]
    piv = piv.reindex(index=tasks, columns=models)

    fig, ax = plt.subplots(figsize=(0.42 * len(models) + 3, 0.5 * len(tasks) + 1.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=10)
    for i in range(len(tasks)):
        for j in range(len(models)):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    ax.set_title(f"Scalar-mix COG (relative depth) — {lang}", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("COG / (n_layers-1)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    df = parse()
    df.to_csv("output/multiling_cog_summary.csv", index=False)
    print(f"parsed {len(df)} rows; models={df.model.nunique()} langs={sorted(df.lang.unique())}")
    print(df.groupby(["lang", "task"]).size().unstack(fill_value=0))
    for lg in ["en", "zh", "tr", "fr", "ru", "de"]:
        heatmap(df, lg, f"plots/figs/multiling_cog_{lg}.png")
