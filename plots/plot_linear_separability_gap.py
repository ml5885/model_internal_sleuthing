import os
import re
from dataclasses import dataclass

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple


sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 22,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,
    }
)


MODEL_NAMES = {
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    "pythia1.4b": "Pythia-1.4B",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "distilbert-base-uncased": "DistilBERT-Base",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-7B-Instruct",
    "olmo2-7b": "OLMo-2-7B",
}


MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "deberta-v3-large",
    "gpt2",
    "gpt2-large",
    "gpt2-xl",
    "qwen2",
    "qwen2-instruct",
    "gemma2b",
    "gemma2b-it",
    "llama3-8b",
    "llama3-8b-instruct",
    "pythia-6.9b",
    "pythia-6.9b-tulu",
    "olmo2-7b-instruct",
    "olmo2-7b",
]


DATASETS = [
    "ud_gum_dataset",
    "ud_zh_gsd_dataset",
    "ud_de_gsd_dataset",
    "ud_fr_gsd_dataset",
    "ud_ru_syntagrus_dataset",
    "ud_tr_imst_dataset",
]


TASKS = ["lexeme", "inflection"]


@dataclass(frozen=True)
class Paths:
    repo_root: str

    @property
    def probes_dir(self) -> str:
        return os.path.join(self.repo_root, "output", "probes")

    @property
    def probes2_dir(self) -> str:
        return os.path.join(self.repo_root, "output", "probes2")


def _repo_root_from_this_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_acc_columns(df: pd.DataFrame, prefix: str) -> Tuple[str, str]:
    """Find (accuracy_col, control_accuracy_col) for various CSV formats."""
    if "Acc" in df.columns and "controlAcc" in df.columns:
        return "Acc", "controlAcc"

    if f"{prefix}_Accuracy" in df.columns and f"{prefix}_ControlAccuracy" in df.columns:
        return f"{prefix}_Accuracy", f"{prefix}_ControlAccuracy"

    for acc_col in df.columns:
        if acc_col.lower() == f"{prefix}_accuracy":
            for ctrl_col in df.columns:
                if ctrl_col.lower() == f"{prefix}_controlaccuracy":
                    return acc_col, ctrl_col

    raise ValueError(
        f"Could not find accuracy columns for prefix='{prefix}'. Available: {list(df.columns)}"
    )


def find_csv_file(
    paths: Paths, dataset: str, model: str, task: str, probe_type: str
) -> Optional[str]:
    probe_type_variants = [probe_type]
    if probe_type == "nn":
        probe_type_variants = ["nn", "mlp", "nonlinear"]
    elif probe_type == "mlp":
        probe_type_variants = ["mlp", "nn", "nonlinear"]
    elif probe_type == "nonlinear":
        probe_type_variants = ["nonlinear", "mlp", "nn"]
    elif probe_type == "reg":
        probe_type_variants = ["reg", "linear"]
    elif probe_type == "linear":
        probe_type_variants = ["linear", "reg"]

    for probe_variant in probe_type_variants:
        probe_dirs = [
            os.path.join(paths.probes_dir, f"{dataset}_{model}_{task}_{probe_variant}"),
            os.path.join(paths.probes2_dir, f"{dataset}_{model}_{task}_{probe_variant}"),
        ]
        for probe_dir in probe_dirs:
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")
            if os.path.exists(csv_path):
                return csv_path

    return None


def _normalize_layers(layers: np.ndarray) -> np.ndarray:
    layers = np.asarray(layers, dtype=float)
    if layers.size == 0:
        return layers
    mn, mx = float(layers.min()), float(layers.max())
    if mx == mn:
        return np.zeros_like(layers, dtype=float)
    return (layers - mn) / (mx - mn)


def _segment_from_norm(layer_norm: float) -> str:
    if layer_norm < (1.0 / 3.0):
        return "early"
    if layer_norm < (2.0 / 3.0):
        return "mid"
    return "late"


def collect_layerwise_gaps(
    datasets: list[str] = DATASETS,
    models: list[str] = MODELS,
    tasks: list[str] = TASKS,
) -> pd.DataFrame:
    """Returns long-form DataFrame with gap data for each layer in both linear and MLP probes."""
    paths = Paths(repo_root=_repo_root_from_this_file())
    rows: list[dict] = []

    for dataset in datasets:
        for model in models:
            for task in tasks:
                lin_csv = find_csv_file(paths, dataset, model, task, "reg")
                mlp_csv = find_csv_file(paths, dataset, model, task, "mlp")

                if not lin_csv or not mlp_csv:
                    continue

                lin_df = pd.read_csv(lin_csv)
                mlp_df = pd.read_csv(mlp_csv)

                lac, lac_ctrl = get_acc_columns(lin_df, task)
                mac, mac_ctrl = get_acc_columns(mlp_df, task)

                common_layers = np.intersect1d(lin_df["Layer"].values, mlp_df["Layer"].values)
                if common_layers.size == 0:
                    continue

                lf = lin_df[lin_df["Layer"].isin(common_layers)].sort_values("Layer")
                mf = mlp_df[mlp_df["Layer"].isin(common_layers)].sort_values("Layer")

                layers = lf["Layer"].values
                layer_norm = _normalize_layers(layers)
                gap = mf[mac].values - lf[lac].values
                lin_selectivity = lf[lac].values - lf[lac_ctrl].values
                mlp_selectivity = mf[mac].values - mf[mac_ctrl].values
                selectivity_gap = mlp_selectivity - lin_selectivity

                for layer, ln, g, sg, lin_sel, mlp_sel in zip(
                    layers, layer_norm, gap, selectivity_gap, lin_selectivity, mlp_selectivity
                ):
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Model": model,
                            "ModelName": MODEL_NAMES.get(model, model),
                            "Task": task,
                            "Layer": float(layer),
                            "LayerNorm": float(ln),
                            "Segment": _segment_from_norm(float(ln)),
                            "Gap": float(g),
                            "SelectivityGap": float(sg),
                            "LinearSelectivity": float(lin_sel),
                            "MLPSelectivity": float(mlp_sel),
                            "LinearCSV": lin_csv,
                            "MLPCSV": mlp_csv,
                        }
                    )

    return pd.DataFrame(rows)


def plot_gap_summary_boxplot(
    layerwise: pd.DataFrame,
    output_dir: str,
    filename: str = "linear_separability_gap_summary.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if layerwise.empty:
        print("[WARN] No data available; skipping figure.")
        return

    unit_df = (
        layerwise.groupby(["Dataset", "Model", "Task", "Segment"], observed=True)
        .agg(
            GapMean=("Gap", "mean"),
            GapMax=("Gap", "max"),
        )
        .reset_index()
    )

    unit_df["Task"] = unit_df["Task"].map({"lexeme": "Lexeme", "inflection": "Inflection"})
    x_order = ["early", "mid", "late"]
    tasks_order = ["Lexeme", "Inflection"]

    y = unit_df["GapMean"].values
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.02, 0.15 * (y_max - y_min if y_max > y_min else 1.0))
    ylim = (y_min - pad, y_max + pad)

    palette = sns.color_palette("Set2", n_colors=2)
    task_to_color = {"Lexeme": palette[0], "Inflection": palette[1]}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 7.0), sharey=True)
    strip_dot_size = 7.5
    strip_dot_alpha = 0.35
    box_linewidth = 2.2
    flier_marker_size = 7.0

    for ax, task_label in zip(axes, tasks_order):
        sub = unit_df[unit_df["Task"] == task_label]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        sns.boxplot(
            data=sub,
            x="Segment",
            y="GapMean",
            order=x_order,
            ax=ax,
            color=task_to_color[task_label],
            width=0.55,
            fliersize=flier_marker_size,
            linewidth=box_linewidth,
            boxprops={"linewidth": box_linewidth},
            whiskerprops={"linewidth": box_linewidth},
            capprops={"linewidth": box_linewidth},
            medianprops={"linewidth": box_linewidth + 0.4},
        )
        sns.stripplot(
            data=sub,
            x="Segment",
            y="GapMean",
            order=x_order,
            ax=ax,
            color="black",
            alpha=strip_dot_alpha,
            size=strip_dot_size,
            jitter=0.18,
        )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(task_label)
        ax.set_xlabel("Layer depth")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("MLP Acc - Lin Acc")
    axes[1].set_ylabel("")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def plot_gap_vs_depth(
    layerwise: pd.DataFrame,
    output_dir: str,
    filename: str = "linear_separability_gap_vs_depth.png",
    n_grid: int = 101,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if layerwise.empty:
        print("[WARN] No data available; skipping depth figure.")
        return

    common_x = np.linspace(0.0, 1.0, n_grid)
    tasks = [("lexeme", "Lexeme"), ("inflection", "Inflection")]
    palette = sns.color_palette("Set2", n_colors=2)
    task_colors = {"Lexeme": palette[0], "Inflection": palette[1]}

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    for task_key, task_label in tasks:
        sub = layerwise[layerwise["Task"] == task_key].copy()
        if sub.empty:
            continue

        # One curve per (dataset, model) unit.
        curves = []
        for (_, _), g in sub.groupby(["Dataset", "Model"], observed=True):
            g = g.sort_values("LayerNorm")
            x = g["LayerNorm"].values
            y = g["Gap"].values
            if len(x) < 2 or np.allclose(x.min(), x.max()):
                continue
            y_interp = np.interp(common_x, x, y)
            curves.append(y_interp)

        if not curves:
            continue

        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        lo = np.quantile(curves, 0.25, axis=0)
        hi = np.quantile(curves, 0.75, axis=0)

        ax.plot(common_x, mean, color=task_colors[task_label], linewidth=2.5, label=task_label)
        ax.fill_between(common_x, lo, hi, color=task_colors[task_label], alpha=0.18, linewidth=0)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("ΔAcc (MLP−Lin)")
    ax.set_title("Gap vs depth", pad=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(title="Task", frameon=True, loc="upper right")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def plot_selectivity_gap_summary_boxplot(
    layerwise: pd.DataFrame,
    output_dir: str,
    filename: str = "selectivity_gap_summary.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if layerwise.empty or "SelectivityGap" not in layerwise.columns:
        print("[WARN] No selectivity data available; skipping figure.")
        return

    unit_df = (
        layerwise.groupby(["Dataset", "Model", "Task", "Segment"], observed=True)
        .agg(
            SelectivityGapMean=("SelectivityGap", "mean"),
            SelectivityGapMax=("SelectivityGap", "max"),
        )
        .reset_index()
    )

    unit_df["Task"] = unit_df["Task"].map({"lexeme": "Lexeme", "inflection": "Inflection"})
    x_order = ["early", "mid", "late"]
    tasks_order = ["Lexeme", "Inflection"]

    y = unit_df["SelectivityGapMean"].values
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.02, 0.15 * (y_max - y_min if y_max > y_min else 1.0))
    ylim = (y_min - pad, y_max + pad)

    palette = sns.color_palette("Set2", n_colors=2)
    task_to_color = {"Lexeme": palette[0], "Inflection": palette[1]}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 7.0), sharey=True)
    strip_dot_size = 7.5
    strip_dot_alpha = 0.35
    box_linewidth = 2.2
    flier_marker_size = 7.0

    for ax, task_label in zip(axes, tasks_order):
        sub = unit_df[unit_df["Task"] == task_label]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        sns.boxplot(
            data=sub,
            x="Segment",
            y="SelectivityGapMean",
            order=x_order,
            ax=ax,
            color=task_to_color[task_label],
            width=0.55,
            fliersize=flier_marker_size,
            linewidth=box_linewidth,
            boxprops={"linewidth": box_linewidth},
            whiskerprops={"linewidth": box_linewidth},
            capprops={"linewidth": box_linewidth},
            medianprops={"linewidth": box_linewidth + 0.4},
        )
        sns.stripplot(
            data=sub,
            x="Segment",
            y="SelectivityGapMean",
            order=x_order,
            ax=ax,
            color="black",
            alpha=strip_dot_alpha,
            size=strip_dot_size,
            jitter=0.18,
        )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(task_label)
        ax.set_xlabel("Layer depth")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("MLP Sel. - Lin Sel.")
    axes[1].set_ylabel("")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def plot_selectivity_gap_vs_depth(
    layerwise: pd.DataFrame,
    output_dir: str,
    filename: str = "selectivity_gap_vs_depth.png",
    n_grid: int = 101,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if layerwise.empty or "SelectivityGap" not in layerwise.columns:
        print("[WARN] No selectivity data available; skipping depth figure.")
        return

    common_x = np.linspace(0.0, 1.0, n_grid)
    tasks = [("lexeme", "Lexeme"), ("inflection", "Inflection")]
    palette = sns.color_palette("Set2", n_colors=2)
    task_colors = {"Lexeme": palette[0], "Inflection": palette[1]}

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    for task_key, task_label in tasks:
        sub = layerwise[layerwise["Task"] == task_key].copy()
        if sub.empty:
            continue

        curves = []
        for (_, _), g in sub.groupby(["Dataset", "Model"], observed=True):
            g = g.sort_values("LayerNorm")
            x = g["LayerNorm"].values
            y = g["SelectivityGap"].values
            if len(x) < 2 or np.allclose(x.min(), x.max()):
                continue
            y_interp = np.interp(common_x, x, y)
            curves.append(y_interp)

        if not curves:
            continue

        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        lo = np.quantile(curves, 0.25, axis=0)
        hi = np.quantile(curves, 0.75, axis=0)

        ax.plot(common_x, mean, color=task_colors[task_label], linewidth=2.5, label=task_label)
        ax.fill_between(common_x, lo, hi, color=task_colors[task_label], alpha=0.18, linewidth=0)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("ΔSelectivity (MLP−Lin)")
    ax.set_title("Selectivity gap vs depth", pad=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(title="Task", frameon=True, loc="upper right")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


MULTILINGUAL_MODEL_NAMES = {
    "byt5": "ByT5",
    "mt5": "mT5",
    "goldfish_eng_latn_1000mb": "Goldfish-EN",
    "goldfish_deu_latn_1000mb": "Goldfish-DE",
    "goldfish_fra_latn_1000mb": "Goldfish-FR",
    "goldfish_rus_cyrl_1000mb": "Goldfish-RU",
    "goldfish_tur_latn_1000mb": "Goldfish-TR",
    "goldfish_zho_hans_1000mb": "Goldfish-ZH",
}

DATASET_DISPLAY_NAMES = {
    "ud_gum_dataset": "English (GUM)",
    "ud_zh_gsd_dataset": "Chinese (GSD)",
    "ud_de_gsd_dataset": "German (GSD)",
    "ud_fr_gsd_dataset": "French (GSD)",
    "ud_ru_syntagrus_dataset": "Russian (SynTagRus)",
    "ud_tr_imst_dataset": "Turkish (IMST)",
}


def get_model_display_name(model: str) -> str:
    if model in MODEL_NAMES:
        return MODEL_NAMES[model]
    if model in MULTILINGUAL_MODEL_NAMES:
        return MULTILINGUAL_MODEL_NAMES[model]
    if model.startswith("goldfish_"):
        return MULTILINGUAL_MODEL_NAMES.get(model, model.replace("_", "-").title())
    return model


def _is_checkpoint_model(model: str) -> bool:
    if "stage1-step" in model or "stage2-step" in model:
        return True
    if re.search(r"_step\d+", model):
        return True
    return False


def discover_models_for_dataset(
    paths: Paths, dataset: str, task: str
) -> list[str]:
    """Find models with both linear and MLP probes. Excludes checkpoints and byt5."""
    models_found = set()

    for probe_dir in [paths.probes_dir, paths.probes2_dir]:
        if not os.path.exists(probe_dir):
            continue

        for entry in os.listdir(probe_dir):
            if not entry.startswith(f"{dataset}_"):
                continue
            if f"_{task}_" not in entry:
                continue

            suffix = entry[len(dataset) + 1:]
            task_pos = suffix.rfind(f"_{task}_")
            if task_pos == -1:
                continue
            model = suffix[:task_pos]
            probe_type = suffix[task_pos + len(task) + 2:]

            if "pca" in probe_type.lower() or probe_type in ["rf"]:
                continue
            if model == "byt5":
                continue
            if _is_checkpoint_model(model):
                continue
            if dataset == "ud_gum_dataset" and (model == "mt5" or model.startswith("goldfish")):
                continue

            if probe_type in ["reg", "linear", "nn", "mlp", "nonlinear"]:
                models_found.add(model)

    valid_models = []
    for model in models_found:
        lin_csv = find_csv_file(paths, dataset, model, task, "reg")
        mlp_csv = find_csv_file(paths, dataset, model, task, "mlp")
        if lin_csv and mlp_csv:
            valid_models.append(model)

    return sorted(valid_models)


def collect_layerwise_gaps_for_dataset(
    dataset: str,
    task: str,
    paths: Optional[Paths] = None,
) -> pd.DataFrame:
    if paths is None:
        paths = Paths(repo_root=_repo_root_from_this_file())
    
    models = discover_models_for_dataset(paths, dataset, task)
    if not models:
        return pd.DataFrame()
    
    rows: list[dict] = []
    
    for model in models:
        lin_csv = find_csv_file(paths, dataset, model, task, "reg")
        mlp_csv = find_csv_file(paths, dataset, model, task, "mlp")
        
        if not lin_csv or not mlp_csv:
            continue
        
        lin_df = pd.read_csv(lin_csv)
        mlp_df = pd.read_csv(mlp_csv)
        
        try:
            lac, lac_ctrl = get_acc_columns(lin_df, task)
            mac, mac_ctrl = get_acc_columns(mlp_df, task)
        except ValueError:
            continue
        
        common_layers = np.intersect1d(lin_df["Layer"].values, mlp_df["Layer"].values)
        if common_layers.size == 0:
            continue
        
        lf = lin_df[lin_df["Layer"].isin(common_layers)].sort_values("Layer")
        mf = mlp_df[mlp_df["Layer"].isin(common_layers)].sort_values("Layer")
        
        layers = lf["Layer"].values
        layer_norm = _normalize_layers(layers)
        lin_selectivity = lf[lac].values - lf[lac_ctrl].values
        mlp_selectivity = mf[mac].values - mf[mac_ctrl].values
        selectivity_gap = mlp_selectivity - lin_selectivity
        
        for layer, ln, sg, lin_sel, mlp_sel in zip(
            layers, layer_norm, selectivity_gap, lin_selectivity, mlp_selectivity
        ):
            rows.append({
                "Dataset": dataset,
                "Model": model,
                "ModelName": get_model_display_name(model),
                "Task": task,
                "Layer": float(layer),
                "LayerNorm": float(ln),
                "SelectivityGap": float(sg),
                "LinearSelectivity": float(lin_sel),
                "MLPSelectivity": float(mlp_sel),
            })
    
    return pd.DataFrame(rows)


def plot_mlp_advantage_multiplot(
    dataset: str,
    task: str,
    output_dir: str,
    n_grid: int = 101,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    df = collect_layerwise_gaps_for_dataset(dataset, task)
    if df.empty:
        print(f"[WARN] No data for {dataset}/{task}; skipping.")
        return
    
    models = df["Model"].unique()
    n_models = len(models)
    
    if n_models == 0:
        print(f"[WARN] No models found for {dataset}/{task}; skipping.")
        return
    
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    elif n_models <= 9:
        n_cols = 3
        n_rows = 3
    elif n_models <= 12:
        n_cols = 4
        n_rows = 3
    elif n_models <= 16:
        n_cols = 4
        n_rows = 4
    elif n_models <= 20:
        n_cols = 5
        n_rows = 4
    else:
        n_cols = 5
        n_rows = (n_models + 4) // 5

    fig_width = 3.5 * n_cols
    fig_height = 3.0 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    if n_models <= 10:
        colors = sns.color_palette("tab10", n_colors=n_models)
    else:
        colors = sns.color_palette("husl", n_colors=n_models)

    common_x = np.linspace(0.0, 1.0, n_grid)
    all_gaps = []
    for model in models:
        sub = df[df["Model"] == model].sort_values("LayerNorm")
        all_gaps.extend(sub["SelectivityGap"].values)
    
    if not all_gaps:
        print(f"[WARN] No gap data for {dataset}/{task}; skipping.")
        plt.close(fig)
        return
    
    y_min, y_max = min(all_gaps), max(all_gaps)
    y_pad = max(0.02, 0.1 * (y_max - y_min))
    y_lim = (y_min - y_pad, y_max + y_pad)

    def model_sort_key(m):
        priority_order = [
            "bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased",
            "deberta-v3-large", "gpt2", "gpt2-large", "gpt2-xl",
            "pythia1.4b", "pythia-6.9b", "pythia-6.9b-tulu",
            "qwen2", "qwen2-instruct", "gemma2b", "gemma2b-it",
            "llama3-8b", "llama3-8b-instruct",
            "olmo2-7b", "olmo2-7b-instruct",
            "byt5", "mt5",
        ]
        for i, p in enumerate(priority_order):
            if m == p or m.startswith(p):
                return (0, i, m)
        if m.startswith("goldfish"):
            return (1, 0, m)
        return (2, 0, m)
    
    sorted_models = sorted(models, key=model_sort_key)
    
    for idx, model in enumerate(sorted_models):
        ax = axes_flat[idx]
        sub = df[df["Model"] == model].sort_values("LayerNorm")
        
        x = sub["LayerNorm"].values
        y = sub["SelectivityGap"].values
        model_name = sub["ModelName"].iloc[0]
        
        color = colors[idx % len(colors)]
        
        ax.plot(x * 100, y, color=color, linewidth=2.0, marker="o", markersize=3, alpha=0.9)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.fill_between(x * 100, 0, y, where=(y > 0), color=color, alpha=0.15, interpolate=True)
        ax.fill_between(x * 100, 0, y, where=(y < 0), color="gray", alpha=0.15, interpolate=True)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(*y_lim)
        ax.set_title(model_name, fontsize=14, fontweight="bold", pad=4)
        ax.grid(True, linestyle="--", alpha=0.3)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Layer Depth (%)", fontsize=14)
        if idx % n_cols == 0:
            ax.set_ylabel("Selectivity Gap", fontsize=14)

        ax.tick_params(axis="both", labelsize=12)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    filename = f"mlp_advantage_{dataset}_{task}.png"
    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main(
    output_dir: str = "figures3",
    datasets: list[str] = DATASETS,
    models: list[str] = MODELS,
) -> None:
    df = collect_layerwise_gaps(datasets=datasets, models=models, tasks=TASKS)
    if df.empty:
        print("[WARN] No matching (linear, mlp) probe pairs found. Nothing to summarize.")
        return

    plot_gap_summary_boxplot(df, output_dir=output_dir)
    plot_gap_vs_depth(df, output_dir=output_dir)
    plot_selectivity_gap_summary_boxplot(df, output_dir=output_dir)
    plot_selectivity_gap_vs_depth(df, output_dir=output_dir)


def main_mlp_advantage_plots(
    output_dir: str = "figures3",
    datasets: list[str] = DATASETS,
) -> None:
    for dataset in datasets:
        for task in TASKS:
            plot_mlp_advantage_multiplot(dataset, task, output_dir=output_dir)


if __name__ == "__main__":
    main()
    main_mlp_advantage_plots()


