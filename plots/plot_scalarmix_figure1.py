"""
Recreate Figure 1 of Tenney et al. (2019), "BERT Rediscovers the Classical NLP
Pipeline".

Per task: F1 columns (baseline P^(0) and full-model P^(L)) on the left, and two
overlaid horizontal bars on the right --
  * light purple = expected layer from cumulative scores (Eq. 4)
  * dark blue    = mixing-weight center of gravity (Eq. 2)

Needs both `cumulative_scores.json` (from --probe_type cumulative) and
`scalarmix_weights.json` (from --probe_type scalarmix).

Usage:
  python -m plots.plot_scalarmix_figure1 --model bert-large-uncased --head linear
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

PURPLE, BLUE = "#c9c3e0", "#4372ad"

TASK_DATASET = {
    "pos": "ud_gum_pos", "constituents": "ud_gum_constituents", "dep": "ud_gum_dep",
    "ner": "ud_gum_ner", "srl": "up_ewt_srl", "coref": "ud_gum_coref_pairs",
    "spr": "spr", "relation": "semeval2010_relations",
}
TASK_LABELS = {
    "pos": "POS", "constituents": "Consts.", "dep": "Deps.", "ner": "Entities",
    "srl": "SRL", "coref": "Coref.", "spr": "SPR", "relation": "Relations",
}
PAPER_ORDER = ["pos", "constituents", "dep", "ner", "srl", "coref", "spr", "relation"]
MODEL_LABELS = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large"}


def _load(root, dataset, model, task, probe, key):
    path = os.path.join(root, f"{dataset}_{model}_{task}_{probe}", f"{key}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def collect(root, model, head):
    """Return {task: dict(expected_layer, cog, base_f1, full_f1, n_layers)}."""
    sm_probe = "scalarmix" if head == "linear" else "scalarmix_mlp"
    cu_probe = "cumulative" if head == "linear" else "cumulative_mlp"
    out = {}
    for task, dataset in TASK_DATASET.items():
        cu = _load(root, dataset, model, task, cu_probe, "cumulative_scores")
        sm = _load(root, dataset, model, task, sm_probe, "scalarmix_weights")
        if cu is None and sm is None:
            continue
        n_layers = (len(cu["f1"]) if cu else sm["n_layers"])
        out[task] = {
            "expected_layer": cu["expected_layer"] if cu else None,
            "base_f1": 100 * cu["baseline_f1"] if cu else None,
            "full_f1": 100 * cu["full_f1"] if cu else None,
            "cog": sm["cog"] if sm else None,
            "n_layers": n_layers,
        }
    return out


def plot(model, data, out_path):
    tasks = [t for t in PAPER_ORDER if t in data]
    n = len(tasks)
    L = max(d["n_layers"] for d in data.values()) - 1

    fig = plt.figure(figsize=(9.5, 0.62 * n + 1.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.2], wspace=0.05)
    ax_f1 = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharey=ax_f1)

    y = np.arange(n)[::-1]  # first task on top

    # ---- right panel: overlaid bars (blue behind = COG, purple front = expected layer) ----
    for yi, t in zip(y, tasks):
        d = data[t]
        if d["cog"] is not None:
            ax.barh(yi, d["cog"], height=0.72, color=BLUE, zorder=2)
            ax.text(d["cog"] - 0.15, yi, f"{d['cog']:.2f}", va="center", ha="right",
                    fontsize=10, color="white", zorder=4)
        if d["expected_layer"] is not None:
            ax.barh(yi, d["expected_layer"], height=0.72, color=PURPLE, zorder=3)
            ax.text(d["expected_layer"] - 0.15, yi, f"{d['expected_layer']:.2f}",
                    va="center", ha="right", fontsize=10, color="black", zorder=4)
    ax.set_xlim(0, L)
    ax.set_xticks(range(0, L + 1, 2))
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_title("Expected layer & center-of-gravity", fontsize=12)
    ax.tick_params(labelleft=False)
    ax.grid(axis="x", color="0.85", lw=0.6)
    ax.set_axisbelow(True)

    # ---- left panel: F1 columns ----
    ax_f1.set_xlim(0, 1)
    ax_f1.axvspan(0, 1, color="0.94", zorder=0)
    for yi, t in zip(y, tasks):
        d = data[t]
        if d["base_f1"] is not None:
            ax_f1.text(0.34, yi, f"{d['base_f1']:.1f}", va="center", ha="center", fontsize=10)
            ax_f1.text(0.72, yi, f"{d['full_f1']:.1f}", va="center", ha="center", fontsize=10)
    ax_f1.text(0.5, 1.10, "F1 Scores", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=12)
    ax_f1.text(0.34, 1.02, r"$\ell{=}0$", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=10, color="0.3")
    ax_f1.text(0.72, 1.02, rf"$\ell{{=}}{L}$", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=10, color="0.3")
    ax_f1.set_yticks(y)
    ax_f1.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=11)
    ax_f1.tick_params(length=0)
    for s in ax_f1.spines.values():
        s.set_visible(False)
    ax_f1.set_xticks([])

    fig.suptitle(f"Summary statistics on {MODEL_LABELS.get(model, model)}", fontsize=13, y=1.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}.png")

    print(f"\n{model} ({'linear' if 'mlp' not in out_path else 'mlp'} head):")
    print(f"  {'task':10s} {'exp_layer':>9s} {'mix_COG':>8s} {'base_F1':>8s} {'full_F1':>8s}")
    for t in tasks:
        d = data[t]
        el = f"{d['expected_layer']:.2f}" if d["expected_layer"] is not None else "  -"
        cg = f"{d['cog']:.2f}" if d["cog"] is not None else "  -"
        bf = f"{d['base_f1']:.1f}" if d["base_f1"] is not None else "  -"
        ff = f"{d['full_f1']:.1f}" if d["full_f1"] is not None else "  -"
        print(f"  {TASK_LABELS[t]:10s} {el:>9s} {cg:>8s} {bf:>8s} {ff:>8s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/probes")
    ap.add_argument("--model", default="bert-large-uncased")
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = collect(args.root, args.model, args.head)
    if not data:
        raise SystemExit(f"No results under {args.root} for {args.model} ({args.head})")
    out = args.out or f"plots/figs/figure1_{args.model}_{args.head}"
    plot(args.model, data, out)


if __name__ == "__main__":
    main()
