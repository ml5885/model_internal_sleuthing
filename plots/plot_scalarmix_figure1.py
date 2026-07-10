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
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "URW Palladio L",
                              "TeX Gyre Pagella", "P052", "Book Antiqua"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

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


def expected_layer(acc):
    """Tenney Eq. 4 on the raw (unclamped) differential of the accuracy curve.
    NaN when the total gain is negligible (task not meaningfully learned)."""
    acc = np.asarray(acc, dtype=float)
    total = acc[-1] - acc[0]
    if total <= 1e-3:
        return np.nan
    return float(np.sum(np.arange(len(acc))[1:] * np.diff(acc)) / total)


def snr(acc):
    """Signal-to-noise of the curve: total gain / back-half layer-to-layer std.
    Low SNR (<~3) means the expected layer is noise-dominated / unreliable."""
    acc = np.asarray(acc, dtype=float)
    noise = np.std(np.diff(acc[len(acc) // 2:]))
    return float((acc[-1] - acc[0]) / noise) if noise > 0 else np.inf


def collect(root, model, head):
    """Return {task: dict(expected_layer, cog, base_f1, full_f1, snr, n_layers)}."""
    sm_probe = "scalarmix" if head == "linear" else "scalarmix_mlp"
    cu_probe = "cumulative" if head == "linear" else "cumulative_mlp"
    out = {}
    for task, dataset in TASK_DATASET.items():
        cu = _load(root, dataset, model, task, cu_probe, "cumulative_scores")
        sm = _load(root, dataset, model, task, sm_probe, "scalarmix_weights")
        if cu is None and sm is None:
            continue
        n_layers = (len(cu["acc"]) if cu else sm["n_layers"])
        out[task] = {
            # recompute from the saved curve so the metric change needs no rerun
            "expected_layer": expected_layer(cu["acc"]) if cu else None,
            "snr": snr(cu["acc"]) if cu else None,
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

    fig = plt.figure(figsize=(6.6, 0.46 * n + 0.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 3.5], wspace=0.03)
    ax_f1 = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharey=ax_f1)

    y = np.arange(n)[::-1]  # first task on top
    BAR_H = 0.82

    # ---- right panel: overlaid bars (blue behind = COG, purple front = expected layer) ----
    for yi, t in zip(y, tasks):
        d = data[t]
        el, cog = d["expected_layer"], d["cog"]
        has_el = el is not None and np.isfinite(el)
        has_cog = cog is not None and np.isfinite(cog)
        if has_cog:
            ax.barh(yi, cog, height=BAR_H, color=BLUE, zorder=2)
        if has_el:
            ax.barh(yi, el, height=BAR_H, color=PURPLE, zorder=3)
        if not (has_el and has_cog):
            for val, txt in [(el if has_el else None, "black"), (cog if has_cog else None, "white")]:
                if val is not None:
                    ax.text(val - 0.25, yi, f"{val:.2f}", va="center", ha="right",
                            fontsize=12, color=txt, fontweight="bold", zorder=5)
            continue
        # Label each value at its bar tip (or just past it if the bar is too short
        # to hold the text). Purple is drawn in front over [0, el] and blue behind
        # over [0, cog], so the colour under any x follows that stacking. Stagger
        # vertically when the two values are within ~2 layers.
        def color_at(x):
            if x <= el + 1e-6:
                return "black"          # on the (front) purple bar
            if x <= cog + 1e-6:
                return "white"          # on the visible blue bar
            return "black"              # on background
        # Stagger vertically when the two labels would crowd. Threshold is wide
        # because a short expected-layer bar puts its label just right of the tip,
        # eating into the gap toward the COG label.
        dy_e, dy_c = (0.17, -0.17) if abs(el - cog) < 3.3 else (0.0, 0.0)
        for val, dy in [(el, dy_e), (cog, dy_c)]:
            if val < 1.8:               # too short: label just past the tip
                x, ha = val + 0.28, "left"
            else:
                x, ha = val - 0.28, "right"
            ax.text(x, yi + dy, f"{val:.2f}", va="center", ha=ha, fontsize=12,
                    color=color_at(x), fontweight="bold", zorder=5)
    ax.set_xlim(0, L)
    ax.set_xticks(range(0, L + 1, 4))
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_title("Expected layer & center-of-gravity", fontsize=13)
    ax.tick_params(labelleft=False, labelsize=12)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    ax.grid(axis="x", color="0.82", lw=0.7)
    ax.set_axisbelow(True)

    # ---- left panel: F1 columns ----
    ax_f1.set_xlim(0, 1)
    ax_f1.axvspan(0, 1, color="0.93", zorder=0)
    for yi, t in zip(y, tasks):
        d = data[t]
        if d["base_f1"] is not None:
            ax_f1.text(0.32, yi, f"{d['base_f1']:.1f}", va="center", ha="center", fontsize=12, fontweight="bold")
            ax_f1.text(0.74, yi, f"{d['full_f1']:.1f}", va="center", ha="center", fontsize=12, fontweight="bold")
    ax_f1.text(0.5, 1.11, "F1 Scores", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax_f1.text(0.32, 1.015, r"$\ell{=}0$", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=11.5, color="0.25")
    ax_f1.text(0.74, 1.015, rf"$\ell{{=}}{L}$", transform=ax_f1.transAxes, ha="center", va="bottom", fontsize=11.5, color="0.25")
    ax_f1.set_yticks(y)
    ax_f1.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=14, fontweight="bold")
    ax_f1.tick_params(length=0)
    for s in ax_f1.spines.values():
        s.set_visible(False)
    ax_f1.set_xticks([])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".png", dpi=300, bbox_inches="tight")
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
