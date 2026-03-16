"""
Statistical significance tests for probing selectivity.

Two tests:
1. McNemar's test  – uses existing predictions.csv files (no retraining).
2. Multi-seed permutation test – re-runs ridge regression with K random
   label permutations to build a null distribution for accuracy.

Both produce per-layer p-values, summary tables, and plots.
"""

import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src import config
from src.train import load_shards, load_layer
from src.probe import solve_ridge


# ── helpers ──────────────────────────────────────────────────────────────────

DATASET_LANG_MAP = {
    "ud_gum_dataset": "en",
    "ud_de_gsd_dataset": "de",
    "ud_fr_gsd_dataset": "fr",
    "ud_ru_syntagrus_dataset": "ru",
    "ud_tr_imst_dataset": "tr",
    "ud_zh_gsd_dataset": "zh",
}


def _extract_dataset_and_model(dirname, task, probe_type):
    """Extract (dataset_key, model_key) from a probe directory basename."""
    base = dirname.rsplit(f"_{task}_{probe_type}", 1)[0]
    for ds_prefix in sorted(DATASET_LANG_MAP.keys(), key=len, reverse=True):
        if base.startswith(ds_prefix + "_"):
            model_key = base[len(ds_prefix) + 1:]
            return ds_prefix, model_key
        if base == ds_prefix:
            return ds_prefix, ""
    return base, ""


def discover_experiments(probes_dir, task, probe_type):
    """Find all experiment directories matching task and probe_type.

    Returns list of (exp_dir, display_name) tuples.
    Skips checkpoint variants (step*, stage*) to keep the main results clean.
    """
    pattern = os.path.join(probes_dir, f"*_{task}_{probe_type}")
    dirs = sorted(glob.glob(pattern))
    results = []
    for d in dirs:
        base = os.path.basename(d)
        # skip checkpoint runs
        if re.search(r"step\d+|stage\d+|tokens\d+", base):
            continue
        # extract model name from directory name pattern: {dataset}_{model}_{task}_{probe}
        parts = base.rsplit(f"_{task}_{probe_type}", 1)[0]
        # dataset is the prefix up to the model key
        results.append((d, parts))
    return results


def get_display_name(exp_name):
    """Convert an experiment name like 'ud_gum_dataset_bert-base-uncased' to a
    readable display name with language tag, e.g. '[en] BERT-Base'."""
    for ds_prefix, lang in sorted(DATASET_LANG_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if exp_name.startswith(ds_prefix + "_"):
            model_key = exp_name[len(ds_prefix) + 1:]
            model_display = config.MODEL_DISPLAY_NAMES.get(model_key, model_key)
            return f"[{lang}] {model_display}"
        if exp_name == ds_prefix:
            return f"[{lang}] (unknown)"
    return exp_name


def load_predictions(exp_dir):
    """Load predictions.csv and return DataFrame."""
    path = os.path.join(exp_dir, "predictions.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


# ── McNemar's test ───────────────────────────────────────────────────────────

def mcnemar_per_layer(pred_df):
    """Compute McNemar's test for each layer.

    For each test sample we know:
      - whether the real-label probe got it right  (y_pred == y_true)
      - whether the control-label probe got it right (y_control_pred == y_control_true)

    McNemar's null hypothesis: the two probes have the same error rate.
    A significant result means the real probe's accuracy is genuinely
    different from (and in our case higher than) the control probe's.

    Returns DataFrame with columns: layer, n, real_acc, ctrl_acc, selectivity,
                                     b, c, chi2, p_value
    """
    rows = []
    for layer, grp in pred_df.groupby("layer"):
        real_correct = (grp["y_pred"] == grp["y_true"]).values
        ctrl_correct = (grp["y_control_pred"] == grp["y_control_true"]).values

        # 2×2 contingency table
        a = int((real_correct & ctrl_correct).sum())
        b = int((real_correct & ~ctrl_correct).sum())   # real right, ctrl wrong
        c = int((~real_correct & ctrl_correct).sum())    # real wrong, ctrl right
        d = int((~real_correct & ~ctrl_correct).sum())

        n = len(grp)
        real_acc = real_correct.mean()
        ctrl_acc = ctrl_correct.mean()

        # Use exact binomial test when counts are small
        table = np.array([[a, b], [c, d]])
        try:
            result = mcnemar(table, exact=(min(b, c) < 25))
            chi2_val = result.statistic
            p_val = result.pvalue
        except Exception:
            chi2_val = np.nan
            p_val = np.nan

        rows.append({
            "layer": int(layer),
            "n": n,
            "real_acc": real_acc,
            "ctrl_acc": ctrl_acc,
            "selectivity": real_acc - ctrl_acc,
            "b_real_only": b,
            "c_ctrl_only": c,
            "statistic": chi2_val,
            "p_value": p_val,
        })

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# ── Permutation test on accuracy difference ──────────────────────────────────

def perm_test_accuracy_diff_per_layer(pred_df, n_permutations=10000, seed=42):
    """Permutation test where the test statistic is the difference in mean
    accuracy (proportion correct) between the real probe and the control probe.

    For each layer:
      1. Compute observed diff = mean(real_correct) - mean(ctrl_correct).
      2. Pool the two binary correct/incorrect vectors.
      3. For each permutation, randomly reassign the "real" vs "control" labels
         and recompute the difference.
      4. p-value = (# permuted diffs >= observed diff + 1) / (K + 1).

    Returns DataFrame with columns: layer, n, real_acc, ctrl_acc, selectivity,
                                     perm_mean_diff, perm_std_diff, p_value
    """
    rows = []
    for layer, grp in pred_df.groupby("layer"):
        real_correct = (grp["y_pred"] == grp["y_true"]).astype(int).values
        ctrl_correct = (grp["y_control_pred"] == grp["y_control_true"]).astype(int).values

        n = len(grp)
        real_acc = real_correct.mean()
        ctrl_acc = ctrl_correct.mean()
        observed_diff = real_acc - ctrl_acc

        # Pool and permute
        rng = np.random.RandomState(seed + int(layer))
        perm_diffs = np.empty(n_permutations)
        pooled = np.concatenate([real_correct, ctrl_correct])

        for i in range(n_permutations):
            perm = rng.permutation(pooled)
            perm_real = perm[:n]
            perm_ctrl = perm[n:]
            perm_diffs[i] = perm_real.mean() - perm_ctrl.mean()

        p_value = (np.sum(perm_diffs >= observed_diff) + 1) / (n_permutations + 1)

        rows.append({
            "layer": int(layer),
            "n": n,
            "real_acc": real_acc,
            "ctrl_acc": ctrl_acc,
            "selectivity": observed_diff,
            "perm_mean_diff": perm_diffs.mean(),
            "perm_std_diff": perm_diffs.std(),
            "p_value": p_value,
        })

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# ── Two-proportion z-test ────────────────────────────────────────────────────

def two_prop_ztest_per_layer(pred_df):
    """Standard two-proportion z-test comparing the accuracy (proportion correct)
    of the real probe vs. the control probe, per layer.

    Applicable when n > 30 (which it always is for these experiments).

    H0: p_real = p_ctrl
    H1: p_real > p_ctrl   (one-sided)

    Test statistic:
        z = (p1 - p2) / sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
    where p_hat is the pooled proportion.

    Returns DataFrame with columns: layer, n, real_acc, ctrl_acc, selectivity,
                                     pooled_p, z_stat, p_value
    """
    rows = []
    for layer, grp in pred_df.groupby("layer"):
        real_correct = (grp["y_pred"] == grp["y_true"]).values
        ctrl_correct = (grp["y_control_pred"] == grp["y_control_true"]).values

        n = len(grp)
        n1 = n  # same samples for both probes
        n2 = n

        p1 = real_correct.mean()
        p2 = ctrl_correct.mean()

        # Pooled proportion
        p_hat = (real_correct.sum() + ctrl_correct.sum()) / (n1 + n2)

        # Avoid division by zero if both are perfect or both are zero
        if p_hat == 0.0 or p_hat == 1.0:
            z_stat = 0.0
            p_value = 1.0
        else:
            se = np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
            z_stat = (p1 - p2) / se
            # One-sided p-value (testing p1 > p2)
            p_value = 1.0 - stats.norm.cdf(z_stat)

        rows.append({
            "layer": int(layer),
            "n": n,
            "real_acc": p1,
            "ctrl_acc": p2,
            "selectivity": p1 - p2,
            "pooled_p": p_hat,
            "z_stat": z_stat,
            "p_value": p_value,
        })

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def run_mcnemar_all(probes_dir, task, probe_types=None):
    """Run McNemar's test across all experiments for a given task.

    Returns a dict mapping (exp_name, probe_type) -> DataFrame of per-layer results.
    """
    if probe_types is None:
        probe_types = ["reg", "nn"]
    all_results = {}
    for pt in probe_types:
        experiments = discover_experiments(probes_dir, task, pt)
        for exp_dir, exp_name in experiments:
            pred_df = load_predictions(exp_dir)
            if pred_df is None:
                continue
            df = mcnemar_per_layer(pred_df)
            df["experiment"] = exp_name
            df["probe_type"] = pt
            all_results[(exp_name, pt)] = df
    return all_results


def run_perm_diff_all(probes_dir, task, probe_types=None, n_permutations=10000, seed=42):
    """Run permutation test (accuracy difference) across all experiments.

    Returns a dict mapping (exp_name, probe_type) -> DataFrame of per-layer results.
    """
    if probe_types is None:
        probe_types = ["reg", "nn"]
    all_results = {}
    for pt in probe_types:
        experiments = discover_experiments(probes_dir, task, pt)
        for exp_dir, exp_name in experiments:
            pred_df = load_predictions(exp_dir)
            if pred_df is None:
                continue
            df = perm_test_accuracy_diff_per_layer(pred_df, n_permutations=n_permutations, seed=seed)
            df["experiment"] = exp_name
            df["probe_type"] = pt
            all_results[(exp_name, pt)] = df
    return all_results


def run_two_prop_ztest_all(probes_dir, task, probe_types=None):
    """Run two-proportion z-test across all experiments.

    Returns a dict mapping (exp_name, probe_type) -> DataFrame of per-layer results.
    """
    if probe_types is None:
        probe_types = ["reg", "nn"]
    all_results = {}
    for pt in probe_types:
        experiments = discover_experiments(probes_dir, task, pt)
        for exp_dir, exp_name in experiments:
            pred_df = load_predictions(exp_dir)
            if pred_df is None:
                continue
            df = two_prop_ztest_per_layer(pred_df)
            df["experiment"] = exp_name
            df["probe_type"] = pt
            all_results[(exp_name, pt)] = df
    return all_results


# ── Multi-seed permutation test ──────────────────────────────────────────────

def permutation_test_layer(X_train, y_train, X_test, y_test,
                           lambda_reg, n_classes, n_permutations=1000,
                           seed=42):
    """Run a permutation test for a single layer using ridge regression.

    1. Compute real accuracy using the true labels.
    2. For each of K permutations, shuffle y_train, re-fit ridge, compute accuracy.
    3. p-value = (# permutation accuracies >= real accuracy + 1) / (K + 1)
       (the +1 is a standard conservative correction)

    Returns dict with real_acc, perm_mean, perm_std, p_value, perm_accs.
    """
    # Real probe
    real_scores = solve_ridge(X_train, y_train, X_test, lambda_reg, n_classes)
    real_acc = (real_scores.argmax(1) == y_test).mean()

    # Permutation null distribution
    rng = np.random.RandomState(seed)
    perm_accs = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_train)
        perm_scores = solve_ridge(X_train, y_perm, X_test, lambda_reg, n_classes)
        perm_accs[i] = (perm_scores.argmax(1) == y_test).mean()

    p_value = (np.sum(perm_accs >= real_acc) + 1) / (n_permutations + 1)

    return {
        "real_acc": real_acc,
        "perm_mean": perm_accs.mean(),
        "perm_std": perm_accs.std(),
        "p_value": p_value,
        "perm_accs": perm_accs,
    }


def run_permutation_test(activations_path, labels_path, task,
                         lambda_reg=1e-3, n_permutations=1000, seed=42):
    """Run permutation test across all layers for a single experiment.

    Returns DataFrame with per-layer results.
    """
    shards = load_shards(activations_path)
    sample = np.load(shards[0], mmap_mode="r")["activations"]
    n_layers = sample.shape[1]

    # Load and prepare labels (mirrors train.py logic)
    df = pd.read_csv(labels_path)

    activations_dir = (activations_path if os.path.isdir(activations_path)
                       else os.path.dirname(activations_path))
    sampled_indices_path = os.path.join(activations_dir, "sampled_indices.csv")
    if os.path.exists(sampled_indices_path):
        sampled_df = pd.read_csv(sampled_indices_path)
        df = df.iloc[sampled_df["index"].values].reset_index(drop=True)

    indices = np.arange(len(df))

    if task in ["lexeme", "inflection"]:
        valid = df["Lemma"].notna()
        if "Inflection Label" in df.columns:
            valid &= df["Inflection Label"].notna()
        df = df[valid].reset_index(drop=True)
        indices = indices[valid]

        lemmas = df["Lemma"].values
        uniq = sorted(set(lemmas))
        lex_labels = np.array([uniq.index(w) for w in lemmas], dtype=int)
        if "Inflection Label" in df.columns:
            infl = df["Inflection Label"].values
            uniq_infl = sorted(set(infl))
            inf_labels = np.array([uniq_infl.index(x) for x in infl], dtype=int)
        else:
            inf_labels = lex_labels
        y_all = lex_labels if task == "lexeme" else inf_labels
    else:
        raise ValueError(f"Permutation test currently supports lexeme/inflection only, got {task}")

    # Filter rare classes (same as train.py)
    true_counts = np.bincount(y_all)
    keep_mask = true_counts[y_all] >= 4
    y_all = y_all[keep_mask]
    indices = indices[keep_mask]

    n_classes = int(y_all.max() + 1)

    rows = []
    all_perm_accs = {}

    for layer_idx in tqdm(range(n_layers), desc="Permutation test layers"):
        X_flat = load_layer(shards, layer_idx)[indices]

        layer_seed = seed + layer_idx
        rng = np.random.RandomState(layer_seed)

        # Same split as probe.py: 70/10/20
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_flat, y_all,
                train_size=config.SPLIT_RATIOS["train"],
                random_state=layer_seed,
                stratify=y_all,
            )
        except ValueError:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_flat, y_all,
                train_size=config.SPLIT_RATIOS["train"],
                random_state=layer_seed,
            )

        val_frac = config.SPLIT_RATIOS["val"] / (
            config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"]
        )
        temp_counts = np.bincount(y_temp)
        strat = y_temp if temp_counts.min() > 1 else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_frac,
            random_state=layer_seed,
            stratify=strat,
        )

        res = permutation_test_layer(
            X_train, y_train, X_test, y_test,
            lambda_reg, n_classes,
            n_permutations=n_permutations,
            seed=layer_seed,
        )

        rows.append({
            "layer": layer_idx,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": n_classes,
            "real_acc": res["real_acc"],
            "perm_mean": res["perm_mean"],
            "perm_std": res["perm_std"],
            "selectivity": res["real_acc"] - res["perm_mean"],
            "p_value": res["p_value"],
        })
        all_perm_accs[layer_idx] = res["perm_accs"]

    result_df = pd.DataFrame(rows)
    return result_df, all_perm_accs


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_mcnemar_summary(all_results, outdir, task):
    """Create summary plots for McNemar's test results."""
    os.makedirs(outdir, exist_ok=True)

    # Collect all results into one DataFrame
    combined = pd.concat(all_results.values(), ignore_index=True)

    # ── Plot 1: p-value heatmap across models and layers ──
    for pt in combined["probe_type"].unique():
        subset = combined[combined["probe_type"] == pt]
        experiments = sorted(subset["experiment"].unique())
        layers_all = sorted(subset["layer"].unique())

        # Build matrix
        pval_matrix = np.full((len(experiments), len(layers_all)), np.nan)
        for i, exp in enumerate(experiments):
            exp_data = subset[subset["experiment"] == exp]
            for _, row in exp_data.iterrows():
                j = layers_all.index(int(row["layer"]))
                pval_matrix[i, j] = row["p_value"]

        # Shorten experiment names for display
        display_names = [get_display_name(exp) for exp in experiments]

        fig, ax = plt.subplots(figsize=(max(12, len(layers_all) * 0.5), max(4, len(experiments) * 0.4)))

        # Log-scale colormap: significant = dark, non-significant = light
        log_pvals = -np.log10(np.clip(pval_matrix, 1e-300, 1.0))
        im = ax.imshow(log_pvals, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=max(10, np.nanmax(log_pvals)))

        # Normalize layer labels to show every nth
        n_labels = len(layers_all)
        step = max(1, n_labels // 15)
        ax.set_xticks(range(0, n_labels, step))
        ax.set_xticklabels([str(layers_all[i]) for i in range(0, n_labels, step)])
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(f"McNemar's Test: {task} ({pt} probe) — $-\\log_{{10}}(p)$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("$-\\log_{10}(p)$")

        # Mark significance thresholds
        for threshold, label in [(3, "p<0.001"), (2, "p<0.01")]:
            if threshold <= np.nanmax(log_pvals):
                cbar.ax.axhline(threshold, color="black", linewidth=0.8, linestyle="--")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"mcnemar_heatmap_{task}_{pt}.png"), dpi=150)
        fig.savefig(os.path.join(outdir, f"mcnemar_heatmap_{task}_{pt}.pdf"))
        plt.close(fig)

    # ── Plot 2: selectivity with significance markers for each model ──
    for pt in combined["probe_type"].unique():
        subset = combined[combined["probe_type"] == pt]
        experiments = sorted(subset["experiment"].unique())

        n_exps = len(experiments)
        ncols = min(4, n_exps)
        nrows = (n_exps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                  squeeze=False)

        for idx, exp in enumerate(experiments):
            ax = axes[idx // ncols][idx % ncols]
            exp_data = subset[subset["experiment"] == exp].sort_values("layer")

            layers = exp_data["layer"].values
            sels = exp_data["selectivity"].values
            pvals = exp_data["p_value"].values

            ax.bar(layers, sels, color="steelblue", alpha=0.7)
            # Mark significant layers
            sig_mask = pvals < 0.001
            ax.bar(layers[sig_mask], sels[sig_mask], color="darkred", alpha=0.8, label="p < 0.001")

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Selectivity")
            ax.set_title(get_display_name(exp), fontsize=9)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(n_exps, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"McNemar's Test: {task} selectivity ({pt} probe)", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"mcnemar_selectivity_{task}_{pt}.png"), dpi=150,
                    bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"mcnemar_selectivity_{task}_{pt}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)


def plot_permutation_results(result_df, perm_accs, outdir, exp_name, task):
    """Plots for a single permutation test experiment."""
    os.makedirs(outdir, exist_ok=True)

    n_layers = len(result_df)

    # ── Plot 1: real accuracy vs permutation null distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = result_df["layer"].values
    ax.fill_between(layers,
                    result_df["perm_mean"] - 2 * result_df["perm_std"],
                    result_df["perm_mean"] + 2 * result_df["perm_std"],
                    alpha=0.3, color="gray", label="Permutation null (mean +/- 2 SD)")
    ax.plot(layers, result_df["perm_mean"], "k--", alpha=0.5, label="Permutation mean")
    ax.plot(layers, result_df["real_acc"], "ro-", markersize=4, label="Real accuracy")

    # Mark significant layers
    sig = result_df["p_value"] < 0.001
    ax.scatter(layers[sig], result_df["real_acc"].values[sig],
               color="darkred", s=50, zorder=5, marker="*", label="p < 0.001")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Permutation Test: {exp_name} — {task}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"permutation_accuracy_{task}.png"), dpi=150)
    fig.savefig(os.path.join(outdir, f"permutation_accuracy_{task}.pdf"))
    plt.close(fig)

    # ── Plot 2: p-value across layers ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(layers, result_df["p_value"].values, "bo-", markersize=4)
    ax.axhline(0.05, color="orange", linestyle="--", label="p = 0.05")
    ax.axhline(0.01, color="red", linestyle="--", label="p = 0.01")
    ax.axhline(0.001, color="darkred", linestyle="--", label="p = 0.001")
    ax.set_xlabel("Layer")
    ax.set_ylabel("p-value")
    ax.set_title(f"Permutation Test p-values: {exp_name} — {task}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"permutation_pvalues_{task}.png"), dpi=150)
    fig.savefig(os.path.join(outdir, f"permutation_pvalues_{task}.pdf"))
    plt.close(fig)

    # ── Plot 3: sample null distributions for early/mid/late layers ──
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set(l for l in sample_layers if l in perm_accs))

    fig, axes = plt.subplots(1, len(sample_layers),
                              figsize=(4 * len(sample_layers), 3.5))
    if len(sample_layers) == 1:
        axes = [axes]

    for ax, l in zip(axes, sample_layers):
        accs = perm_accs[l]
        real = result_df.loc[result_df["layer"] == l, "real_acc"].values[0]
        pval = result_df.loc[result_df["layer"] == l, "p_value"].values[0]

        ax.hist(accs, bins=40, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(real, color="red", linewidth=2, label=f"Real acc = {real:.3f}")
        ax.set_title(f"Layer {l} (p = {pval:.1e})", fontsize=9)
        ax.set_xlabel("Accuracy")
        ax.legend(fontsize=7)

    fig.suptitle(f"Null distributions: {exp_name} — {task}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"permutation_nulldist_{task}.png"), dpi=150)
    fig.savefig(os.path.join(outdir, f"permutation_nulldist_{task}.pdf"))
    plt.close(fig)


# ── Tables ───────────────────────────────────────────────────────────────────

def print_mcnemar_table(all_results, task):
    """Print a summary table to console and return LaTeX string."""
    rows = []
    for (exp_name, pt), df in sorted(all_results.items()):
        display = get_display_name(exp_name)
        n_layers = len(df)
        n_sig = (df["p_value"] < 0.001).sum()
        mean_sel = df["selectivity"].mean()
        min_p = df["p_value"].min()
        max_p = df["p_value"].max()
        rows.append({
            "Model": display,
            "Probe": pt,
            "Layers": n_layers,
            "Sig (p<.001)": f"{n_sig}/{n_layers}",
            "Mean Sel.": f"{mean_sel:.4f}",
            "Min p": f"{min_p:.2e}",
            "Max p": f"{max_p:.2e}",
        })

    table_df = pd.DataFrame(rows)
    print(f"\n{'='*80}")
    print(f"McNemar's Test Summary — {task}")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))
    print()

    # LaTeX
    latex = table_df.to_latex(index=False, escape=True,
                               caption=f"McNemar's test summary for {task} probing.",
                               label=f"tab:mcnemar_{task}")
    return table_df, latex


def print_generic_test_table(all_results, task, test_name):
    """Print a summary table for any test that produces per-layer p-values.

    Works for perm_diff, two_prop_ztest, or any test with the same structure.
    Returns (table_df, latex_string).
    """
    rows = []
    for (exp_name, pt), df in sorted(all_results.items()):
        display = get_display_name(exp_name)
        n_layers = len(df)
        n_sig = (df["p_value"] < 0.001).sum()
        mean_sel = df["selectivity"].mean()
        min_p = df["p_value"].min()
        max_p = df["p_value"].max()
        rows.append({
            "Model": display,
            "Probe": pt,
            "Layers": n_layers,
            "Sig (p<.001)": f"{n_sig}/{n_layers}",
            "Mean Sel.": f"{mean_sel:.4f}",
            "Min p": f"{min_p:.2e}",
            "Max p": f"{max_p:.2e}",
        })

    table_df = pd.DataFrame(rows)
    print(f"\n{'='*80}")
    print(f"{test_name} Summary — {task}")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))
    print()

    latex = table_df.to_latex(index=False, escape=True,
                               caption=f"{test_name} summary for {task} probing.",
                               label=f"tab:{test_name.lower().replace(' ', '_')}_{task}")
    return table_df, latex


def plot_generic_test_summary(all_results, outdir, task, test_name, file_prefix):
    """Create heatmap and selectivity plots for any test with per-layer p-values.

    Reuses the same plotting logic as plot_mcnemar_summary.
    """
    os.makedirs(outdir, exist_ok=True)

    combined = pd.concat(all_results.values(), ignore_index=True)

    for pt in combined["probe_type"].unique():
        subset = combined[combined["probe_type"] == pt]
        experiments = sorted(subset["experiment"].unique())
        layers_all = sorted(subset["layer"].unique())

        # Build p-value matrix
        pval_matrix = np.full((len(experiments), len(layers_all)), np.nan)
        for i, exp in enumerate(experiments):
            exp_data = subset[subset["experiment"] == exp]
            for _, row in exp_data.iterrows():
                j = layers_all.index(int(row["layer"]))
                pval_matrix[i, j] = row["p_value"]

        display_names = [get_display_name(exp) for exp in experiments]

        # ── Heatmap ──
        fig, ax = plt.subplots(figsize=(max(12, len(layers_all) * 0.5),
                                         max(4, len(experiments) * 0.4)))
        log_pvals = -np.log10(np.clip(pval_matrix, 1e-300, 1.0))
        im = ax.imshow(log_pvals, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=max(10, np.nanmax(log_pvals)))

        n_labels = len(layers_all)
        step = max(1, n_labels // 15)
        ax.set_xticks(range(0, n_labels, step))
        ax.set_xticklabels([str(layers_all[i]) for i in range(0, n_labels, step)])
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(f"{test_name}: {task} ({pt} probe) — $-\\log_{{10}}(p)$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("$-\\log_{10}(p)$")
        for threshold, label in [(3, "p<0.001"), (2, "p<0.01")]:
            if threshold <= np.nanmax(log_pvals):
                cbar.ax.axhline(threshold, color="black", linewidth=0.8, linestyle="--")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{file_prefix}_heatmap_{task}_{pt}.png"), dpi=150)
        fig.savefig(os.path.join(outdir, f"{file_prefix}_heatmap_{task}_{pt}.pdf"))
        plt.close(fig)

        # ── Selectivity bar plots ──
        n_exps = len(experiments)
        ncols = min(4, n_exps)
        nrows = (n_exps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                  squeeze=False)

        for idx, exp in enumerate(experiments):
            ax = axes[idx // ncols][idx % ncols]
            exp_data = subset[subset["experiment"] == exp].sort_values("layer")
            layers = exp_data["layer"].values
            sels = exp_data["selectivity"].values
            pvals = exp_data["p_value"].values

            ax.bar(layers, sels, color="steelblue", alpha=0.7)
            sig_mask = pvals < 0.001
            ax.bar(layers[sig_mask], sels[sig_mask], color="darkred", alpha=0.8, label="p < 0.001")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Selectivity")
            ax.set_title(get_display_name(exp), fontsize=9)
            if idx == 0:
                ax.legend(fontsize=7)

        for idx in range(n_exps, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{test_name}: {task} selectivity ({pt} probe)", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{file_prefix}_selectivity_{task}_{pt}.png"), dpi=150,
                    bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"{file_prefix}_selectivity_{task}_{pt}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)


def print_permutation_table(result_df, exp_name, task):
    """Print permutation test results table."""
    display_df = result_df[["layer", "n_test", "real_acc", "perm_mean", "perm_std",
                             "selectivity", "p_value"]].copy()
    display_df.columns = ["Layer", "N_test", "Real Acc", "Perm Mean", "Perm SD",
                           "Selectivity", "p-value"]
    for col in ["Real Acc", "Perm Mean", "Perm SD", "Selectivity"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    display_df["p-value"] = display_df["p-value"].map(lambda x: f"{x:.2e}")

    print(f"\n{'='*80}")
    print(f"Permutation Test — {exp_name} — {task}")
    print(f"{'='*80}")
    print(display_df.to_string(index=False))
    print()

    n_sig = (result_df["p_value"] < 0.001).sum()
    print(f"Significant layers (p < 0.001): {n_sig}/{len(result_df)}")
    print(f"All layers significant (p < 0.05): {(result_df['p_value'] < 0.05).all()}")
    print()


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate_permutation_results(results_dir, task, outdir):
    """Collect all permutation test results into a summary table and heatmap.

    Expects subdirectories named permutation_{exp_name}_{task}/ each containing
    permutation_results_{task}.csv.
    """
    os.makedirs(outdir, exist_ok=True)

    pattern = os.path.join(results_dir, f"permutation_*_{task}",
                            f"permutation_results_{task}.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        # Also try without task suffix in dir name
        pattern = os.path.join(results_dir, "permutation_*",
                                f"permutation_results_{task}.csv")
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"No permutation results found in {results_dir} for task={task}")
        return

    all_dfs = []
    summary_rows = []
    for f in files:
        dirn = os.path.basename(os.path.dirname(f))
        # Extract experiment name: permutation_{exp_name}_{task} -> exp_name
        exp_name = dirn.replace(f"permutation_", "").replace(f"_{task}", "")

        df = pd.read_csv(f)
        df["experiment"] = exp_name
        all_dfs.append(df)

        n_layers = len(df)
        n_sig = (df["p_value"] < 0.001).sum()
        summary_rows.append({
            "Model": config.MODEL_DISPLAY_NAMES.get(exp_name, exp_name),
            "Layers": n_layers,
            "Sig (p<.001)": f"{n_sig}/{n_layers}",
            "Mean Sel.": f"{df['selectivity'].mean():.4f}",
            "Mean Real Acc": f"{df['real_acc'].mean():.4f}",
            "Mean Perm Acc": f"{df['perm_mean'].mean():.4f}",
            "Min p": f"{df['p_value'].min():.2e}",
            "Max p": f"{df['p_value'].max():.2e}",
        })

    combined = pd.concat(all_dfs, ignore_index=True)

    # Print summary
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*90}")
    print(f"Permutation Test Summary — {task} (K=1000)")
    print(f"{'='*90}")
    print(summary_df.to_string(index=False))
    print()

    # Overall statistics
    total_layers = len(combined)
    total_sig = (combined["p_value"] < 0.001).sum()
    print(f"Overall: {total_sig}/{total_layers} layer-model pairs significant at p < 0.001")
    print(f"Overall: {(combined['p_value'] < 0.05).sum()}/{total_layers} significant at p < 0.05")
    print()

    # Save
    summary_df.to_csv(os.path.join(outdir, f"permutation_summary_{task}.csv"), index=False)
    latex = summary_df.to_latex(index=False, escape=True,
                                 caption=f"Permutation test summary for {task} probing (K=1000).",
                                 label=f"tab:permutation_{task}")
    with open(os.path.join(outdir, f"permutation_summary_{task}.tex"), "w") as f:
        f.write(latex)

    # ── Heatmap ──
    experiments = sorted(combined["experiment"].unique())
    layers_all = sorted(combined["layer"].unique())

    pval_matrix = np.full((len(experiments), len(layers_all)), np.nan)
    for i, exp in enumerate(experiments):
        exp_data = combined[combined["experiment"] == exp]
        for _, row in exp_data.iterrows():
            j = layers_all.index(int(row["layer"]))
            pval_matrix[i, j] = row["p_value"]

    display_names = [config.MODEL_DISPLAY_NAMES.get(e, e) for e in experiments]

    fig, ax = plt.subplots(figsize=(max(12, len(layers_all) * 0.5),
                                     max(4, len(experiments) * 0.45)))
    log_pvals = -np.log10(np.clip(pval_matrix, 1e-300, 1.0))
    im = ax.imshow(log_pvals, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(5, np.nanmax(log_pvals)))

    n_labels = len(layers_all)
    step = max(1, n_labels // 15)
    ax.set_xticks(range(0, n_labels, step))
    ax.set_xticklabels([str(layers_all[i]) for i in range(0, n_labels, step)])
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title(f"Permutation Test: {task} — $-\\log_{{10}}(p)$ (K=1000)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$-\\log_{10}(p)$")
    for threshold in [2, 3]:
        if threshold <= np.nanmax(log_pvals):
            cbar.ax.axhline(threshold, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"permutation_heatmap_{task}.png"), dpi=150)
    fig.savefig(os.path.join(outdir, f"permutation_heatmap_{task}.pdf"))
    plt.close(fig)

    print(f"Aggregate results saved to {outdir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for probing selectivity")
    sub = parser.add_subparsers(dest="command")

    # McNemar's test
    mc = sub.add_parser("mcnemar", help="McNemar's test on existing predictions")
    mc.add_argument("--probes_dir", default=os.path.join(config.OUTPUT_DIR, "probes"),
                    help="Directory containing probe output directories")
    mc.add_argument("--task", required=True, choices=["lexeme", "inflection"])
    mc.add_argument("--probe_types", nargs="+", default=["reg", "nn"],
                    help="Probe types to include")
    mc.add_argument("--outdir", default=None,
                    help="Output directory for plots/tables")

    # Permutation test
    pm = sub.add_parser("permutation", help="Multi-seed permutation test")
    pm.add_argument("--activations", required=True,
                    help="Path to activations directory or .npz file")
    pm.add_argument("--labels", required=True, help="Path to labels CSV")
    pm.add_argument("--task", required=True, choices=["lexeme", "inflection"])
    pm.add_argument("--exp_name", default="experiment",
                    help="Experiment label for output")
    pm.add_argument("--lambda_reg", type=float, default=1e-3)
    pm.add_argument("--n_permutations", type=int, default=1000)
    pm.add_argument("--outdir", default=None)

    # Permutation test on accuracy difference (uses predictions.csv)
    pd_cmd = sub.add_parser("perm_diff",
                             help="Permutation test (diff in means) on existing predictions")
    pd_cmd.add_argument("--probes_dir", default=os.path.join(config.OUTPUT_DIR, "probes"),
                         help="Directory containing probe output directories")
    pd_cmd.add_argument("--task", required=True, choices=["lexeme", "inflection"])
    pd_cmd.add_argument("--probe_types", nargs="+", default=["reg", "nn"],
                         help="Probe types to include")
    pd_cmd.add_argument("--n_permutations", type=int, default=10000,
                         help="Number of permutations (default: 10000)")
    pd_cmd.add_argument("--outdir", default=None,
                         help="Output directory for plots/tables")

    # Two-proportion z-test (uses predictions.csv)
    zp = sub.add_parser("ztest",
                         help="Two-proportion z-test on existing predictions")
    zp.add_argument("--probes_dir", default=os.path.join(config.OUTPUT_DIR, "probes"),
                     help="Directory containing probe output directories")
    zp.add_argument("--task", required=True, choices=["lexeme", "inflection"])
    zp.add_argument("--probe_types", nargs="+", default=["reg", "nn"],
                     help="Probe types to include")
    zp.add_argument("--outdir", default=None,
                     help="Output directory for plots/tables")

    # Aggregate permutation results
    ag = sub.add_parser("aggregate", help="Aggregate permutation test results into summary")
    ag.add_argument("--results_dir",
                    default=os.path.join(config.OUTPUT_DIR, "statistical_tests"),
                    help="Directory containing permutation_* subdirectories")
    ag.add_argument("--task", required=True, choices=["lexeme", "inflection"])
    ag.add_argument("--outdir", default=None)

    args = parser.parse_args()

    if args.command == "mcnemar":
        outdir = args.outdir or os.path.join(config.OUTPUT_DIR, "statistical_tests",
                                              f"mcnemar_{args.task}")
        os.makedirs(outdir, exist_ok=True)

        all_results = run_mcnemar_all(args.probes_dir, args.task, args.probe_types)

        if not all_results:
            print(f"No experiments found for task={args.task}")
            return

        table_df, latex = print_mcnemar_table(all_results, args.task)
        plot_mcnemar_summary(all_results, outdir, args.task)

        # Save CSV and LaTeX
        table_df.to_csv(os.path.join(outdir, f"mcnemar_summary_{args.task}.csv"), index=False)
        with open(os.path.join(outdir, f"mcnemar_summary_{args.task}.tex"), "w") as f:
            f.write(latex)

        # Save per-experiment detail CSVs
        for (exp_name, pt), df in all_results.items():
            safe_name = exp_name.replace("/", "_")
            df.to_csv(os.path.join(outdir, f"mcnemar_detail_{safe_name}_{pt}.csv"), index=False)

        print(f"Results saved to {outdir}")

    elif args.command == "permutation":
        outdir = args.outdir or os.path.join(config.OUTPUT_DIR, "statistical_tests",
                                              f"permutation_{args.exp_name}_{args.task}")
        os.makedirs(outdir, exist_ok=True)

        result_df, perm_accs = run_permutation_test(
            args.activations, args.labels, args.task,
            lambda_reg=args.lambda_reg,
            n_permutations=args.n_permutations,
        )

        print_permutation_table(result_df, args.exp_name, args.task)
        plot_permutation_results(result_df, perm_accs, outdir, args.exp_name, args.task)

        result_df.to_csv(os.path.join(outdir, f"permutation_results_{args.task}.csv"), index=False)

        # Save null distributions for reproducibility
        np.savez_compressed(
            os.path.join(outdir, f"permutation_null_dists_{args.task}.npz"),
            **{f"layer_{k}": v for k, v in perm_accs.items()},
        )

        print(f"Results saved to {outdir}")

    elif args.command == "perm_diff":
        outdir = args.outdir or os.path.join(config.OUTPUT_DIR, "statistical_tests",
                                              f"perm_diff_{args.task}")
        os.makedirs(outdir, exist_ok=True)

        all_results = run_perm_diff_all(args.probes_dir, args.task, args.probe_types,
                                         n_permutations=args.n_permutations)

        if not all_results:
            print(f"No experiments found for task={args.task}")
            return

        table_df, latex = print_generic_test_table(all_results, args.task,
                                                    "Permutation Test (Acc Diff)")
        plot_generic_test_summary(all_results, outdir, args.task,
                                   "Permutation Test (Acc Diff)", "perm_diff")

        table_df.to_csv(os.path.join(outdir, f"perm_diff_summary_{args.task}.csv"), index=False)
        with open(os.path.join(outdir, f"perm_diff_summary_{args.task}.tex"), "w") as f:
            f.write(latex)

        for (exp_name, pt), df in all_results.items():
            safe_name = exp_name.replace("/", "_")
            df.to_csv(os.path.join(outdir, f"perm_diff_detail_{safe_name}_{pt}.csv"), index=False)

        print(f"Results saved to {outdir}")

    elif args.command == "ztest":
        outdir = args.outdir or os.path.join(config.OUTPUT_DIR, "statistical_tests",
                                              f"ztest_{args.task}")
        os.makedirs(outdir, exist_ok=True)

        all_results = run_two_prop_ztest_all(args.probes_dir, args.task, args.probe_types)

        if not all_results:
            print(f"No experiments found for task={args.task}")
            return

        table_df, latex = print_generic_test_table(all_results, args.task,
                                                    "Two-Proportion Z-Test")
        plot_generic_test_summary(all_results, outdir, args.task,
                                   "Two-Proportion Z-Test", "ztest")

        table_df.to_csv(os.path.join(outdir, f"ztest_summary_{args.task}.csv"), index=False)
        with open(os.path.join(outdir, f"ztest_summary_{args.task}.tex"), "w") as f:
            f.write(latex)

        for (exp_name, pt), df in all_results.items():
            safe_name = exp_name.replace("/", "_")
            df.to_csv(os.path.join(outdir, f"ztest_detail_{safe_name}_{pt}.csv"), index=False)

        print(f"Results saved to {outdir}")

    elif args.command == "aggregate":
        outdir = args.outdir or os.path.join(config.OUTPUT_DIR, "statistical_tests",
                                              f"aggregate_{args.task}")
        os.makedirs(outdir, exist_ok=True)

        aggregate_permutation_results(args.results_dir, args.task, outdir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
