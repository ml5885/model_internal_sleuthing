"""
Side-by-side comparison of Tenney et al. (2019) Figure 1 (BERT-large) against our
scalar-mix + cumulative probes on the fixed (token-aligned) activations.

Light purple = expected layer (cumulative scores, Eq. 4);
dark blue     = mixing-weight center of gravity (Eq. 2).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

PURPLE, BLUE = "#c9c3e0", "#4372ad"

# Paper Figure 1, BERT-large (24 layers). (expected_layer, mixing_cog)
PAPER = {
    "POS": (3.39, 11.68), "Consts.": (3.79, 13.06), "Deps.": (5.69, 13.75),
    "Entities": (4.64, 13.16), "SRL": (6.54, 13.63), "Coref.": (9.47, 15.80),
    "SPR": (9.93, 12.72), "Relations": (9.40, 12.83),
}
ORDER = ["POS", "Consts.", "Deps.", "Entities", "SRL", "Coref.", "SPR", "Relations"]

TASK_DATASET = {
    "POS": ("pos", "ud_gum_pos"), "Consts.": ("constituents", "ud_gum_constituents"),
    "Deps.": ("dep", "ud_gum_dep"), "Entities": ("ner", "ud_gum_ner"),
    "SRL": ("srl", "up_ewt_srl"), "Coref.": ("coref", "ud_gum_coref_pairs"),
    "SPR": ("spr", "spr"), "Relations": ("relation", "semeval2010_relations"),
}


def load_ours(root, model):
    out = {}
    for name, (task, ds) in TASK_DATASET.items():
        cu = os.path.join(root, f"{ds}_{model}_{task}_cumulative", "cumulative_scores.json")
        sm = os.path.join(root, f"{ds}_{model}_{task}_scalarmix", "scalarmix_weights.json")
        el = json.load(open(cu))["expected_layer"] if os.path.exists(cu) else np.nan
        cog = json.load(open(sm))["cog"] if os.path.exists(sm) else np.nan
        out[name] = (el, cog)
    return out


def draw_panel(ax, data, title, L):
    y = np.arange(len(ORDER))[::-1]
    for yi, t in zip(y, ORDER):
        el, cog = data[t]
        # blue (cog) behind, purple (expected layer) in front
        pairs = sorted([(cog, BLUE, "white"), (el, PURPLE, "black")], key=lambda p: -p[0])
        for val, color, _ in pairs:
            if np.isfinite(val):
                ax.barh(yi, val, height=0.72, color=color, zorder=2 if color == BLUE else 3)
        for val, _, txtc in pairs:
            if np.isfinite(val):
                ax.text(val - 0.2, yi, f"{val:.2f}", va="center", ha="right",
                        fontsize=9.5, color=txtc, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(ORDER, fontsize=11)
    ax.set_xlim(0, L)
    ax.set_xticks(range(0, L + 1, 2))
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlabel("Expected layer & center-of-gravity", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="0.85", lw=0.6)
    ax.set_axisbelow(True)


def main():
    root, model, L = "output/probes_fixed", "bert-large-uncased", 24
    ours = load_ours(root, model)

    fig, (axp, axo) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    draw_panel(axp, PAPER, "Tenney et al. 2019 (BERT-large)", L)
    draw_panel(axo, ours, "Ours, fixed activations (BERT-large)", L)
    axo.tick_params(labelleft=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PURPLE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["Expected layer (cumulative, Eq. 4)", "Mixing-weight COG (Eq. 2)"],
               loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04))

    out = "plots/figs/figure1_comparison_bert-large"
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {out}.png")

    print(f"\n{'task':10s} {'paper_exp':>9s} {'ours_exp':>9s} | {'paper_cog':>9s} {'ours_cog':>9s}")
    for t in ORDER:
        print(f"{t:10s} {PAPER[t][0]:9.2f} {ours[t][0]:9.2f} | {PAPER[t][1]:9.2f} {ours[t][1]:9.2f}")


if __name__ == "__main__":
    main()
