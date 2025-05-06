from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "tokenizer_name": "gpt2",
    },
    "pythia1.4b": {
        "model_name": "EleutherAI/pythia-1.4b",
        "tokenizer_name": "EleutherAI/pythia-1.4b",
    },
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
    },
    "qwen2": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B",
    },
    # "qwen2-instruct": {
    #     "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    #     "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
    # },
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
    },
    "bert-large-uncased": {
        "model_name": "bert-large-uncased",
        "tokenizer_name": "bert-large-uncased",
    },
    # "distilbert-base-uncased": {
    #     "model_name": "distilbert-base-uncased",
    #     "tokenizer_name": "distilbert-base-uncased",
    # },
    "deberta-v3-large": {
        "model_name": "microsoft/deberta-v3-large",
        "tokenizer_name": "microsoft/deberta-v3-large",
    },
}

TESTS = [
    ("king", "man", "woman", "queen"),
    ("man", "king", "queen", "woman"),
    ("walked", "walk", "jump", "jumped"),
    ("go", "went", "run", "ran"),
    ("sang", "sing", "ring", "rang"),
    ("sing", "sang", "rang", "ring"),
]

def get_embedding(tokenizer, embeddings: torch.Tensor, word: str, method: str) -> torch.Tensor:
    """Embedding by subtoken averaging (tokenize) or sum (sum)."""
    if method == "tokenize":
        toks = tokenizer.tokenize(word, add_special_tokens=False)
    else:
        toks = tokenizer.tokenize(" " + word, add_special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(toks)
    vecs = embeddings[ids]
    return vecs.mean(0) if method == "tokenize" else vecs.sum(0)


def get_word_rank(tokenizer, embeddings: torch.Tensor, query_vec: torch.Tensor,
                  word: str, method: str) -> int:
    """Return 1-based cosine-similarity rank of the expected word."""
    emb_norm = F.normalize(embeddings, dim=1)
    q_norm = F.normalize(query_vec.unsqueeze(0), dim=1)
    sims = torch.mm(q_norm, emb_norm.T).squeeze(0)

    if method == "tokenize":
        tok_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_special_tokens=False))
        tok_ids = [tok_ids[-1]] # Get last token
    else:
        tok_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(" " + word, add_special_tokens=False)
        )

    sorted_idx = torch.argsort(sims, descending=True)
    ranks = [(sorted_idx == tid).nonzero(as_tuple=True)[0].item() + 1 for tid in tok_ids]
    return int(sum(ranks) / len(ranks))

def run_models(model_keys: List[str]) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []

    for key in model_keys:
        print(f"Loading {key}...", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained(MODEL_CONFIGS[key]["tokenizer_name"])
        mod = AutoModel.from_pretrained(MODEL_CONFIGS[key]["model_name"]).to(device).eval()

        with torch.no_grad():
            emb = mod.get_input_embeddings().weight.data.to(device)

            for a, b, c, d in TESTS:
                for method in ("tokenize", "sum"):
                    va = get_embedding(tok, emb, a, method)
                    vb = get_embedding(tok, emb, b, method)
                    vc = get_embedding(tok, emb, c, method)
                    query = va - vb + vc
                    rank = get_word_rank(tok, emb, query, d, method)
                    records.append(
                        {"model": key,
                         "analogy": f"{a}-{b}+{c}->{d}",
                         "method": method,
                         "rank": rank}
                    )

        del mod, emb
        torch.cuda.empty_cache()

    return pd.DataFrame.from_records(records)

def make_plots(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    sns.set_theme(style="white", palette="pastel")
    outdir.mkdir(parents=True, exist_ok=True)

    scatter_df = (
        df.pivot_table(index=["model", "analogy"], columns="method", values="rank")
        .dropna()
        .reset_index()
    )

    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_df,
        x="sum",
        y="tokenize",
        hue="model",
        style="analogy",
        palette="pastel",
        ax=ax_s,
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )

    max_rank = scatter_df[["sum", "tokenize"]].to_numpy().max()
    ax_s.plot([1, max_rank], [1, max_rank], ls="--", lw=1, color="grey")
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.tick_params(axis="x", labelsize=16)
    ax_s.tick_params(axis="y", labelsize=16)
    ax_s.set_xlabel("No tokenization (subtoken-sum) - rank of correct word")
    ax_s.set_ylabel("Apply tokenization (subtoken-average) - rank of correct word")
    ax_s.set_title("Tokenization effect on analogy-completion rank")
    ax_s.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig_s.tight_layout()
    scatter_path = outdir / "tokenize_vs_sum_scatter.png"
    fig_s.savefig(scatter_path, dpi=300)
    print(f"Saved {scatter_path}")

    bar_df = (
        scatter_df.assign(delta=lambda x: x["tokenize"] - x["sum"])
        .groupby("model", as_index=False)["delta"]
        .mean()
        .sort_values("delta", ascending=False)
    )

    fig_b, ax_b = plt.subplots(figsize=(0.5 * len(bar_df) + 3, 4))
    sns.barplot(
        data=bar_df,
        x="model",
        y="delta",
        hue="model",
        palette="pastel",
        legend=False,
        ax=ax_b,
    )
    ax_b.axhline(0, color="grey", linewidth=0.8)
    ax_b.set_ylabel("Mean rank difference (tokenize - sum)")
    ax_b.set_xlabel("")
    ax_b.set_title("Average Impact of Tokenization on Analogy Rank")

    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=30, ha="right")

    fig_b.tight_layout()
    delta_path = outdir / "delta_rank_bar.png"
    fig_b.savefig(delta_path, dpi=300)
    print(f"Saved {delta_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate analogy ranks and create scatter + bar plots."
    )
    p.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=f"Subset to evaluate (choices: {', '.join(MODEL_CONFIGS)})"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every model in the registry (overrides --models)."
    )
    p.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=".",
        help="Directory to save figures (default: current directory)."
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    chosen = list(MODEL_CONFIGS) if args.all or not args.models else [
        m for m in args.models if m in MODEL_CONFIGS
    ]
    if not chosen:
        sys.exit("No valid models specified.")

    df = run_models(chosen)
    make_plots(df, args.outdir)

if __name__ == "__main__":
    main()
