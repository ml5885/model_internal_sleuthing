"""
Cross-model Figure-1 (Tenney et al.) multiplot: one figure per language, a grid
of per-model panels, each in the figure1_fixed_bert-base.png style -- F1 columns
(baseline / full-model accuracy) plus nested expected-layer (purple, cumulative)
and mixing-weight center-of-gravity (blue, scalar mix) bars, on that model's
absolute layer axis.

Reads output/probes_multiling/{dataset_id}_{model}_{task}_{scalarmix,cumulative}.
Panels render COG immediately; expected-layer + F1 fill in as cumulative completes.
"""
import glob
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"]
plt.rcParams["font.weight"] = "bold"
PURPLE, BLUE = "#c9c3e0", "#4372ad"

MODELS = ["bert-base-uncased", "bert-large-uncased", "deberta-v3-large", "gpt2", "gpt2-large",
          "gpt2-xl", "pythia-6.9b", "pythia-6.9b-tulu", "olmo2-7b", "olmo2-7b-instruct",
          "gemma2b", "gemma2b-it", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct",
          "llama3-8b", "llama3-8b-instruct", "mt5", "goldfish_eng_latn_1000mb",
          "goldfish_zho_hans_1000mb", "goldfish_deu_latn_1000mb", "goldfish_fra_latn_1000mb",
          "goldfish_rus_cyrl_1000mb", "goldfish_tur_latn_1000mb"]
MODEL_LABEL = {m: m.replace("_latn_1000mb", "").replace("_hans_1000mb", "").replace("_cyrl_1000mb", "")
               .replace("goldfish_", "goldfish-").replace("-uncased", "").replace("-instruct", "-it")
               for m in MODELS}
TASKS = ["pos", "constituents", "dep", "ner", "srl", "coref", "relation"]
TASK_LABEL = {"pos": "POS", "constituents": "Consts.", "dep": "Deps.", "ner": "Entities",
              "srl": "SRL", "coref": "Coref.", "relation": "Relations"}
DS_LANG = {"ud_gum": "en", "up_ewt": "en", "semeval2010": "en", "ud_chinese": "zh", "msra": "zh",
           "up_chinese": "zh", "duie": "zh", "ud_turkish": "tr", "turkish_wikiner": "tr",
           "corefud_turkish": "tr", "ud_french": "fr", "wikiner_fr": "fr", "up_french": "fr",
           "corefud_french": "fr", "redfm_fr": "fr", "ud_russian": "ru", "nerel": "ru",
           "rucoco": "ru", "ud_german": "de", "germeval": "de", "up_german": "de",
           "corefud_german": "de", "redfm_de": "de"}


def _lang(ds):
    return next((lg for p, lg in DS_LANG.items() if ds.startswith(p)), "?")


def _expected_layer(acc):
    acc = np.asarray(acc, float)
    tot = acc[-1] - acc[0]
    return float(np.sum(np.arange(len(acc))[1:] * np.diff(acc)) / tot) if tot > 1e-3 else np.nan


def collect(root):
    data = {}
    for d in glob.glob(os.path.join(root, "*_scalarmix")) + glob.glob(os.path.join(root, "*_cumulative")):
        kind = "cumulative" if d.endswith("_cumulative") else "scalarmix"
        base = os.path.basename(d)[:-(len(kind) + 1)]
        task = base.rsplit("_", 1)[1]
        if task not in TASKS:
            continue
        rest = base[:-(len(task) + 1)]
        model = next((m for m in sorted(MODELS, key=len, reverse=True) if rest.endswith("_" + m)), None)
        if not model:
            continue
        lang = _lang(rest[:-(len(model) + 1)])
        jf = os.path.join(d, f"{'scalarmix_weights' if kind == 'scalarmix' else 'cumulative_scores'}.json")
        if not os.path.exists(jf):
            continue
        rec = data.setdefault((lang, model, task), {})
        j = json.load(open(jf))
        if kind == "scalarmix":
            rec["cog"] = j["cog"]
            rec["n_layers"] = j["n_layers"]
        else:
            rec["expected"] = _expected_layer(j["acc"])
            rec["base_f1"] = 100 * j["baseline_acc"]
            rec["full_f1"] = 100 * j["full_acc"]
            rec["n_layers"] = len(j["acc"])
    return data


def draw_panel(ax_f1, ax, model, recs, L, show_task_labels):
    tasks = [t for t in TASKS if t in recs]
    y = np.arange(len(tasks))[::-1]
    W = 0.13 * L                                     # ~label width in layer units
    for yi, t in zip(y, tasks):
        r = recs[t]
        el, cog = r.get("expected"), r.get("cog")
        has_el = el is not None and np.isfinite(el)
        has_cog = cog is not None and np.isfinite(cog)
        if has_cog:
            ax.barh(yi, cog, height=0.8, color=BLUE, zorder=2)
        if has_el:
            ax.barh(yi, el, height=0.8, color=PURPLE, zorder=3)
        if has_el and has_cog:                       # both -> bert-figure label style
            if el < W + 0.3:                         # purple too short: label past its tip (on blue)
                ax.text(el + 0.12, yi, f"{el:.1f}", va="center", ha="left", fontsize=6,
                        color="white", fontweight="bold", zorder=5)
            else:
                ax.text(el - 0.15, yi, f"{el:.1f}", va="center", ha="right", fontsize=6,
                        color="black", fontweight="bold", zorder=5)
            if (el < W + 0.3) or (cog - el) < W + 0.4:   # crowded: cog outside the blue tip
                ax.text(cog + 0.12, yi, f"{cog:.1f}", va="center", ha="left", fontsize=6,
                        color="black", fontweight="bold", zorder=5)
            else:
                ax.text(cog - 0.15, yi, f"{cog:.1f}", va="center", ha="right", fontsize=6,
                        color="white", fontweight="bold", zorder=5)
        elif has_cog:
            ax.text(cog - 0.15, yi, f"{cog:.1f}", va="center", ha="right", fontsize=6,
                    color="white", fontweight="bold", zorder=5)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.5, len(tasks) - 0.5)
    ax.set_xticks([0, L])
    ax.set_yticks([])
    ax.tick_params(labelsize=6, length=2)
    ax.set_title(MODEL_LABEL.get(model, model), fontsize=8, pad=2)
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)

    # F1 columns (baseline / full accuracy). Task labels only on the leftmost panel.
    ax_f1.set_xlim(0, 1)
    ax_f1.set_ylim(-0.5, len(tasks) - 0.5)
    ax_f1.axvspan(0, 1, color="0.94", zorder=0)
    ax_f1.set_xticks([])
    for s in ax_f1.spines.values():
        s.set_visible(False)
    ax_f1.tick_params(length=0)
    if show_task_labels:
        ax_f1.set_yticks(y)
        ax_f1.set_yticklabels([TASK_LABEL[t] for t in tasks], fontsize=7)
    else:
        ax_f1.set_yticks([])
    for yi, t in zip(y, tasks):
        r = recs[t]
        if r.get("base_f1") is not None:
            ax_f1.text(0.3, yi, f"{r['base_f1']:.0f}", va="center", ha="center", fontsize=6)
            ax_f1.text(0.75, yi, f"{r['full_f1']:.0f}", va="center", ha="center", fontsize=6)


def plot_lang(data, lang, out):
    models = [m for m in MODELS if any(l == lang and mo == m for (l, mo, _) in data)]
    if not models:
        return
    n = len(models)
    ncol = min(5, n)
    nrow = math.ceil(n / ncol)
    fig = plt.figure(figsize=(3.3 * ncol, 1.9 * nrow))
    gs = fig.add_gridspec(nrow, ncol * 2, width_ratios=[0.5, 2.6] * ncol, wspace=0.08, hspace=0.5)
    for k, m in enumerate(models):
        recs = {t: data[(lang, m, t)] for (l, mo, t) in data if l == lang and mo == m}
        L = max((r.get("n_layers", 13) for r in recs.values()), default=13) - 1
        r0, c0 = k // ncol, (k % ncol) * 2
        draw_panel(fig.add_subplot(gs[r0, c0]), fig.add_subplot(gs[r0, c0 + 1]), m, recs, L,
                   show_task_labels=(k % ncol == 0))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"saved {out}  ({n} models)")


if __name__ == "__main__":
    data = collect("output/probes_multiling")
    n_exp = sum("expected" in r for r in data.values())
    print(f"{len(data)} (lang,model,task) cells; {n_exp} with expected-layer")
    for lg in ["en", "zh", "tr", "fr", "ru", "de"]:
        plot_lang(data, lg, f"plots/figs/figure1_multiling_{lg}.png")
