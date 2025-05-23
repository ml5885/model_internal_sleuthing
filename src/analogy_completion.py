import argparse
import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
from src import config

hf_logging.set_verbosity_error()

models = config.ANALOGY_MODEL_LIST
model_names = config.MODEL_DISPLAY_NAMES

palette_map = dict(zip(models, sns.color_palette("pastel", len(models))))

TESTS = [
    ("king", "man", "woman", "queen"),
    ("man", "king", "queen", "woman"),
    ("walked", "walk", "jump", "jumped"),
    ("go", "went", "run", "ran"),
    ("sang", "sing", "ring", "rang"),
    ("sing", "sang", "rang", "ring"),
]

def get_embedding(tokenizer, embeddings, word, method):
    prefix = "" if method == "tokenize" else " "
    toks = tokenizer.tokenize(prefix + word, add_special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(toks)
    vecs = embeddings[ids]
    return vecs.mean(0) if method == "tokenize" else vecs.sum(0)

def get_word_rank(tokenizer, embeddings, query_vec, word, method):
    emb_norm = F.normalize(embeddings, dim=1)
    q_norm = F.normalize(query_vec.unsqueeze(0), dim=1)
    sims = torch.mm(q_norm, emb_norm.T).squeeze(0)

    prefix = "" if method == "tokenize" else " "
    toks = tokenizer.tokenize(prefix + word, add_special_tokens=False)
    tok_ids = tokenizer.convert_tokens_to_ids(toks)
    if method == "tokenize":
        tok_ids = tok_ids[-1:]

    sorted_idx = torch.argsort(sims, descending=True)
    ranks = [(sorted_idx == tid).nonzero(as_tuple=True)[0].item() + 1 for tid in tok_ids]
    return int(sum(ranks) / len(ranks))

def run_models(keys: List[str]) -> pd.DataFrame:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_list = [f"cuda:{i}" for i in range(n_gpus)]
        print(f"Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}", file=sys.stderr)
    else:
        device_list = ["cpu"]
        print("Using CPU", file=sys.stderr)
    recs: list[dict] = []
    for idx, k in enumerate(keys):
        device = device_list[idx % len(device_list)]
        print(f"Loading {k} on {device}...", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained(k)
        mod = AutoModel.from_pretrained(k).to(device).eval()
        with torch.no_grad():
            emb = mod.get_input_embeddings().weight.data.to(device)
            for a, b, c, d in TESTS:
                for method in ("tokenize", "sum"):
                    va = get_embedding(tok, emb, a, method)
                    vb = get_embedding(tok, emb, b, method)
                    vc = get_embedding(tok, emb, c, method)
                    qv = va - vb + vc
                    recs.append({
                        "model": k,
                        "analogy": f"{a}-{b}+{c}->{d}",
                        "method": method,
                        "rank": get_word_rank(tok, emb, qv, d, method),
                    })
        del mod, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame.from_records(recs)

def run_or_load_model(key: str, outdir: pathlib.Path) -> pd.DataFrame:
    safe_key = key.replace("/", "-")
    model_dir = outdir / safe_key
    model_dir.mkdir(parents=True, exist_ok=True)
    path_subdir = model_dir / "results.csv"
    path_flat = outdir / f"{safe_key}_results.csv"
    if path_subdir.exists():
        return pd.read_csv(path_subdir)
    elif path_flat.exists():
        return pd.read_csv(path_flat)
    df = run_models([key])
    df.to_csv(path_subdir, index=False)
    return df

def make_plots(df: pd.DataFrame, outdir: pathlib.Path):
    sns.set_style("white")
    outdir.mkdir(parents=True, exist_ok=True)

    sd = (
        df.pivot_table(index=["model", "analogy"], columns="method", values="rank")
          .dropna()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 9))
    ana_list = [f"{a}-{b}+{c}->{d}" for a, b, c, d in TESTS]
    markers = ["o", "X", "s", "P", "D", "*"]

    sns.scatterplot(
        data=sd,
        x="sum",
        y="tokenize",
        hue="model",
        hue_order=models,
        palette=palette_map,
        style="analogy",
        style_order=ana_list,
        markers=markers,
        s=100,
        edgecolor="black",
        linewidth=1,
        alpha=0.9,
        ax=ax,
        legend=False,
    )

    mx = sd[["sum", "tokenize"]].to_numpy().max()
    ax.plot([1, mx], [1, mx], ls="--", lw=1, color="gray", zorder=0)

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel(
        "No tokenization (subtoken-sum) - rank of correct word",
        fontsize=14,
        labelpad=16,
    )
    ax.set_ylabel(
        "Apply tokenization (subtoken-average) - rank of correct word",
        fontsize=14,
        labelpad=6,
    )
    ax.tick_params(axis="x", labelsize=14, length=4, width=1)
    ax.tick_params(axis="y", labelsize=14, length=4, width=1)
    for sp in ax.spines.values():
        sp.set_linewidth(1)

    # Only show model legend on the plot
    model_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=palette_map[m],
            markeredgecolor="black",
            linestyle="",
            markersize=7,
        )
        for m in models
    ]
    model_labels = [model_names[m] for m in models]

    # Analogy legend handles
    ana_handles = [
        Line2D(
            [0], [0],
            marker=markers[i],
            color="black",
            linestyle="",
            markersize=10,
        )
        for i in range(len(ana_list))
    ]
    ana_labels = ana_list
    # Model legend (lower right, on plot)
    model_legend = ax.legend(
        model_handles,
        model_labels,
        loc="lower right",
        bbox_to_anchor=(1, 0),  # moved slightly down
        frameon=True,
        framealpha=0.7,
        fontsize=12,
        handletextpad=0.3,
        labelspacing=0.3,
        borderpad=0.4,
        ncol=1,
        title="Models",
        title_fontsize=12,
    )
    ax.add_artist(model_legend)

    # Analogy legend (below the plot, under x-axis, spanning full plot width)
    analogy_legend = plt.legend(
        ana_handles,
        ana_labels,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.375),
        frameon=True,
        framealpha=0.7,
        fontsize=11,
        handletextpad=0.3,
        labelspacing=0.9,
        borderpad=0.7,
        ncol=3,
        title="Analogies",
        title_fontsize=11,
        markerscale=0.8,
    )
    ax.add_artist(analogy_legend)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.38)  # increase bottom margin for legend
    scatter_path = outdir / "tokenize_vs_sum_scatter.png"
    fig.savefig(
        scatter_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[model_legend, analogy_legend]
    )
    print(f"Saved {scatter_path}")

    bar_df = (
        sd.assign(delta=lambda x: x["tokenize"] - x["sum"])
          .groupby("model", as_index=False)["delta"].mean()
          .set_index("model")
          .loc[models]
          .reset_index()
    )
    fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
    sns.barplot(
        data=bar_df,
        x="model",
        y="delta",
        hue="model",
        hue_order=models,
        palette=palette_map,
        dodge=False,
        ax=ax_b,
        legend=False,
    )
    ax_b.axhline(0, color="gray", linewidth=1)
    ax_b.set_ylabel("Mean rank difference (tokenize - sum)", fontsize=11)
    ax_b.set_xlabel("")
    ax_b.tick_params(axis="x", labelsize=9, length=4, width=1)
    ax_b.tick_params(axis="y", labelsize=9, length=4, width=1)
    plt.setp(ax_b.get_xticklabels(), rotation=30, ha="right")
    for sp in ax_b.spines.values():
        sp.set_linewidth(1.2)

    fig_b.tight_layout()
    delta_path = outdir / "delta_rank_bar.png"
    fig_b.savefig(delta_path, dpi=300)
    print(f"Saved {delta_path}")

def generate_mock_data():
    import numpy as np
    rng = np.random.default_rng(42)
    ana_list = [f"{a}-{b}+{c}->{d}" for a, b, c, d in TESTS]
    return pd.DataFrame([
        {"model": m, "analogy": ana, "method": meth, "rank": int(rng.integers(1, 100))}
        for m in models for ana in ana_list for meth in ("tokenize", "sum")
    ])

def main():
    p = argparse.ArgumentParser(description="Evaluate analogy ranks and plot.")
    p.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=f"Subset (choices: {', '.join(models)})",
    )
    p.add_argument("--all", action="store_true", help="Evaluate all models.")
    p.add_argument(
        "--outdir",
        type=pathlib.Path,
        default="notebooks/figures5",
        help="Directory for figures.",
    )
    p.add_argument("--mock", action="store_true", help="Use mock data.")
    args = p.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    chosen = models if args.all or not args.models else [
        m for m in args.models if m in models
    ]
    if not chosen:
        sys.exit("No valid models specified.")

    if args.mock:
        df = generate_mock_data()
        df.to_csv(outdir / "mock_data.csv", index=False)
        print("Saved mock data.")
        make_plots(df, outdir)
    else:
        dfs = []
        for m in chosen:
            df = run_or_load_model(m, outdir)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        make_plots(df, outdir)

if __name__ == "__main__":
    main()
