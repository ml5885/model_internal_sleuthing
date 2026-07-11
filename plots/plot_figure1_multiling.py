"""
Cross-model Figure-1 (Tenney et al.) multiplot: one figure per language, a grid
of per-model panels. Each panel shows, per task, the expected layer (purple,
cumulative) and the mixing-weight center of gravity (blue, scalar mix), on that
model's absolute layer axis -- the same style as figure1_fixed_bert-base.png.

Reads output/probes_multiling/{dataset_id}_{model}_{task}_{scalarmix,cumulative}.
Panels render COG immediately; expected-layer fills in as cumulative completes.
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
    if tot <= 1e-3:
        return np.nan
    return float(np.sum(np.arange(len(acc))[1:] * np.diff(acc)) / tot)


def collect(root):
    data = {}  # (lang, model, task) -> dict
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
        if kind == "scalarmix":
            w = json.load(open(jf))
            rec["cog"] = w["cog"]
            rec["n_layers"] = w["n_layers"]
        else:
            c = json.load(open(jf))
            rec["expected"] = _expected_layer(c["acc"])
            rec["full_acc"] = 100 * c["full_acc"]
            rec["n_layers"] = len(c["acc"])
    return data


def panel(ax, model, recs, L):
    tasks = [t for t in TASKS if t in recs]
    y = np.arange(len(tasks))[::-1]
    for yi, t in zip(y, tasks):
        r = recs[t]
        cog, el = r.get("cog"), r.get("expected")
        if cog is not None and np.isfinite(cog):
            ax.barh(yi, cog, height=0.8, color=BLUE, zorder=2)
        if el is not None and np.isfinite(el):
            ax.barh(yi, el, height=0.8, color=PURPLE, zorder=3)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.5, len(tasks) - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABEL[t] for t in tasks], fontsize=8)
    ax.set_xticks([0, L])
    ax.tick_params(labelsize=7, length=2)
    ax.set_title(MODEL_LABEL.get(model, model), fontsize=9)
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)


def plot_lang(data, lang, out):
    models = [m for m in MODELS if (lang, m) in {(l, mo) for (l, mo, _) in data}]
    if not models:
        return
    n = len(models)
    ncol = min(5, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.7 * nrow + 0.6), squeeze=False)
    for k, m in enumerate(models):
        recs = {t: data[(lang, m, t)] for (l, mo, t) in data if l == lang and mo == m}
        L = max((r.get("n_layers", 2) for r in recs.values()), default=13) - 1
        panel(axes[k // ncol][k % ncol], m, recs, L)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    has_exp = any("expected" in r for r in data.values() if True)
    fig.suptitle(f"Expected layer (purple) & center-of-gravity (blue) — {lang}"
                 + ("" if has_exp else "  [COG only; cumulative running]"), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}  ({n} models)")


if __name__ == "__main__":
    data = collect("output/probes_multiling")
    n_exp = sum("expected" in r for r in data.values())
    print(f"{len(data)} (lang,model,task) cells; {n_exp} with expected-layer (cumulative)")
    for lg in ["en", "zh", "tr", "fr", "ru", "de"]:
        plot_lang(data, lg, f"plots/figs/figure1_multiling_{lg}.png")
