import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 100
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 32,
    "legend.title_fontsize": 22,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0
})

bbox_to_anchor = (0, -0.11, 1, 0.1)

model_names = {
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "pythia1.4b": "Pythia-1.4B",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
    "pythia-6.9b_step1": "Pythia-6.9B (step1)",
    "pythia-6.9b_step64": "Pythia-6.9B (step64)",
    "pythia-6.9b_step6000": "Pythia-6.9B (step6k)",
    "pythia-6.9b_step19000": "Pythia-6.9B (step19k)",
    "pythia-6.9b_step37000": "Pythia-6.9B (step37k)",
    "pythia-6.9b_step57000": "Pythia-6.9B (step57k)",
    "pythia-6.9b_step82000": "Pythia-6.9B (step82k)",
    "pythia-6.9b_step111000": "Pythia-6.9B (step111k)",
    "olmo2-7b_stage1-step5000-tokens21B": "OLMo2-7B (5k, 21B tokens)",
    "olmo2-7b_stage1-step40000-tokens168B": "OLMo2-7B (40k, 168B tokens)",
    "olmo2-7b_stage1-step97000-tokens407B": "OLMo2-7B (97k, 407B tokens)",
    "olmo2-7b_stage1-step179000-tokens751B": "OLMo2-7B (179k, 751B tokens)",
    "olmo2-7b_stage1-step282000-tokens1183B": "OLMo2-7B (282k, 1183B tokens)",
    "olmo2-7b_stage1-step409000-tokens1716B": "OLMo2-7B (409k, 1716B tokens)",
    "olmo2-7b_stage1-step559000-tokens2345B": "OLMo2-7B (559k, 2345B tokens)",
    "olmo2-7b_stage1-step734000-tokens3079B": "OLMo2-7B (734k, 3079B tokens)",
}

MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#a55194", "#393b79",
    "#637939", "#e6550d", "#9ecae1", "#fd8d3c"
]

def get_model_color(model, model_list):
    if ("pythia-6.9b" in model and ("step" in model or model == "pythia-6.9b")) or \
       ("olmo2-7b" in model and ("stage1-step" in model or model == "olmo2-7b")):
        if "pythia-6.9b" in model:
            return get_checkpoint_gradient_color(model, model_list, "#2ca02c")
        elif "olmo2-7b" in model:
            return get_checkpoint_gradient_color(model, model_list, "#1f77b4")

    idx = model_list.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

def get_checkpoint_gradient_color(model, model_list, base_color="#1f77b4"):
    if "pythia-6.9b" in model:
        checkpoint_models = [m for m in model_list if "pythia-6.9b" in m]
        base_name = "pythia-6.9b"
    elif "olmo2-7b" in model:
        checkpoint_models = [m for m in model_list if "olmo2-7b" in m and "stage1-step" in m]
        if "olmo2-7b" in model_list:
            checkpoint_models = ["olmo2-7b"] + checkpoint_models
        base_name = "olmo2-7b"
    else:
        idx = model_list.index(model)
        return MODEL_COLORS[idx % len(MODEL_COLORS)]
    
    if len(checkpoint_models) <= 1:
        return base_color
    
    def extract_step_number(model_name):
        if model_name == base_name:
            return float('inf')
        if "step" in model_name:
            try:
                if "tokens" in model_name:
                    tokens_part = model_name.split("tokens")[1].split("B")[0]
                    return float(tokens_part)
                else:
                    step_part = model_name.split("step")[1].split("-")[0].split("_")[0]
                    if "k" in step_part:
                        return float(step_part.replace("k", "")) * 1000
                    return float(step_part)
            except:
                return 0
        return 0
    
    sorted_checkpoints = sorted(checkpoint_models, key=extract_step_number)
    
    try:
        model_idx = sorted_checkpoints.index(model)
    except ValueError:
        return base_color
    
    n_models = len(sorted_checkpoints)
    if n_models == 1:
        return base_color
    
    viridis = plt.cm.viridis
    color_position = 0.2 + (0.7 * model_idx / (n_models - 1))
    rgba_color = viridis(color_position)
    
    return mcolors.to_hex(rgba_color)

def extract_checkpoint_progress(model_name: str, base_name: str) -> float:
    if model_name == base_name:
        return float("inf")
    if "tokens" in model_name:
        return float(model_name.split("tokens")[1].split("B")[0])
    if "step" in model_name:
        step_part = model_name.split("step")[1].split("-")[0].split("_")[0]
        return float(step_part.replace("k", "")) * 1000 if "k" in step_part else float(step_part)
    return 0.0

def plot_checkpoint_training_deltas_by_depth(
    dataset: str,
    olmo_models: list[str],
    pythia_models: list[str],
    probe: str = "reg",
    output_dir: str = "figures3",
    save_name: str = "olmo_pythia_training_deltas_by_depth",
    pca: bool = False,
    pca_dim: int = 50,
    filename = None,
):
    panels = [
        ("inflection", "accuracy"),
        ("inflection", "selectivity"),
        ("lexeme", "accuracy"),
        ("lexeme", "selectivity"),
    ]

    family_specs = [
        ("OLMo-2-7B", olmo_models, "olmo2-7b", "#1f77b4"),
        ("Pythia-6.9B", pythia_models, "pythia-6.9b", "#2ca02c"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = np.atleast_2d(axes)

    acc_deltas_all: list[np.ndarray] = []
    sel_deltas_all: list[np.ndarray] = []

    deltas: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for family_name, models, base_name, _color in family_specs:
        earliest_model = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))[0]
        final_model = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))[-1]
        for task in ["inflection", "lexeme"]:
            probe_dir_early = os.path.join("..", "output", "probes", f"{dataset}_{earliest_model}_{task}_{probe}")
            probe_dir_final = os.path.join("..", "output", "probes", f"{dataset}_{final_model}_{task}_{probe}")
            if pca:
                probe_dir_early += f"_pca_{pca_dim}"
                probe_dir_final += f"_pca_{pca_dim}"
            csv_early = os.path.join(probe_dir_early, f"{task}_results.csv")
            csv_final = os.path.join(probe_dir_final, f"{task}_results.csv")
            if not (os.path.exists(csv_early) and os.path.exists(csv_final)):
                print(f"[WARN] Missing results for {family_name} ({task})")
                continue
            df_early = pd.read_csv(csv_early)
            df_final = pd.read_csv(csv_final)
            acc_col, ctrl_col = get_acc_columns(df_final, task)
            merged = df_early.merge(df_final, on="Layer", suffixes=("_early", "_final"))
            merged = merged.sort_values("Layer").reset_index(drop=True)
            merged["Layer_Normalized"] = (merged["Layer"] - merged["Layer"].min()) / (merged["Layer"].max() - merged["Layer"].min())

            acc_delta = merged[f"{acc_col}_final"] - merged[f"{acc_col}_early"]
            sel_delta = (merged[f"{acc_col}_final"] - merged[f"{ctrl_col}_final"]) - (merged[f"{acc_col}_early"] - merged[f"{ctrl_col}_early"])

            x = merged["Layer_Normalized"].to_numpy()
            deltas[(family_name, task, "accuracy")] = (x, acc_delta.to_numpy())
            deltas[(family_name, task, "selectivity")] = (x, sel_delta.to_numpy())
            acc_deltas_all.append(acc_delta.to_numpy())
            sel_deltas_all.append(sel_delta.to_numpy())

    max_abs_acc = float(np.max(np.abs(np.concatenate(acc_deltas_all)))) if acc_deltas_all else 0.01
    max_abs_sel = float(np.max(np.abs(np.concatenate(sel_deltas_all)))) if sel_deltas_all else 0.01
    acc_ylim = (-1.05 * max_abs_acc, 1.05 * max_abs_acc)
    sel_ylim = (-1.05 * max_abs_sel, 1.05 * max_abs_sel)

    for row, (family_name, _models, _base_name, color) in enumerate(family_specs):
        for col, (task, metric) in enumerate(panels):
            ax = axes[row, col]
            key = (family_name, task, metric)
            if key in deltas:
                x, y = deltas[key]
                ax.plot(x, y, linewidth=3.0, color=color)

            if row == 0:
                ax.set_title("Accuracy" if metric == "accuracy" else "Selectivity", fontsize=20, pad=10)
            
            ax.tick_params(axis="both", which="major", length=8, width=1.8)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            
            if row == 1:
                ax.set_xticklabels(["0", "25", "50", "75", "100"])
            else:
                ax.set_xticklabels([])
            
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

            if metric == "accuracy":
                ax.set_ylim(*acc_ylim)
            else:
                ax.set_ylim(*sel_ylim)

            if col == 0:
                ax.set_ylabel("Last - first checkpoint", fontsize=16)

    # Row labels on the left
    fig.text(0.008, 0.75, "OLMo-2-7B", rotation=90, va="center", ha="center", fontsize=22)
    fig.text(0.008, 0.28, "Pythia-6.9B", rotation=90, va="center", ha="center", fontsize=22)

    # Column group labels at top
    fig.text(0.27, 0.98, "Inflection", ha="center", va="bottom", fontsize=24)
    fig.text(0.77, 0.98, "Lexeme", ha="center", va="bottom", fontsize=24)

    # Common x-axis label
    fig.text(
        0.5,
        0.02,
        "Normalized layer depth (%)",
        ha="center",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"{save_name}{'_pca_' + str(pca_dim) if pca else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved training delta-by-depth figure to {out_path}")

def plot_checkpoint_vs_training_progress_depth_bands(
    dataset: str,
    olmo_models: list[str],
    pythia_models: list[str],
    probe: str = "reg",
    output_dir: str = "figures3",
    save_name: str = "olmo_pythia_training_progress_depth_bands",
    pca: bool = False,
    pca_dim: int = 50,
    filename = None,
    early_band: tuple[float, float] = (0.0, 0.2),
    middle_band: tuple[float, float] = (0.4, 0.6),
    deep_band: tuple[float, float] = (0.8, 1.0),
):
    panels = [
        ("inflection", "accuracy", "Inflection"),
        ("inflection", "selectivity", "Inflection"),
        ("lexeme", "accuracy", "Lexeme"),
        ("lexeme", "selectivity", "Lexeme"),
    ]

    bands = [
        ("0-20%", early_band, 0.35),
        ("40-60%", middle_band, 0.70),
        ("80-100%", deep_band, 1.00),
    ]

    family_specs = [
        ("OLMo-2-7B", olmo_models, "olmo2-7b", "#1f77b4"),
        ("Pythia-6.9B", pythia_models, "pythia-6.9b", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    def _shade(base_hex: str, strength: float) -> str:
        base_rgb = np.array(mcolors.to_rgb(base_hex))
        strength = float(np.clip(strength, 0.0, 1.0))
        rgb = 1.0 - (1.0 - base_rgb) * strength  # 0 -> white, 1 -> base
        return mcolors.to_hex(rgb)

    def training_progress_percent(model_name: str, base_name: str, min_v: float, max_v: float) -> float:
        v = extract_checkpoint_progress(model_name, base_name)
        if v == float("inf"):
            v = max_v
        if max_v == min_v:
            return 100.0
        return 100.0 * (v - min_v) / (max_v - min_v)

    legend_handles = []
    for ax, (task, metric, title) in zip(axes, panels):
        for family_name, models, base_name, color in family_specs:
            ordered_models = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))
            raw = [extract_checkpoint_progress(m, base_name) for m in ordered_models]
            finite = [v for v in raw if v != float("inf")]
            min_v = min(finite) if finite else 0.0
            max_v = max(finite) if finite else 1.0
            x = np.array([training_progress_percent(m, base_name, min_v, max_v) for m in ordered_models])

            for band_label, (lo, hi), shade_strength in bands:
                ys = []
                for model in ordered_models:
                    probe_dir = os.path.join("..", "output", "probes", f"{dataset}_{model}_{task}_{probe}")
                    if pca:
                        probe_dir += f"_pca_{pca_dim}"
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    df = pd.read_csv(csv_path)
                    acc_col, ctrl_col = get_acc_columns(df, task)
                    df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                    band_df = df[(df["Layer_Normalized"] >= lo) & (df["Layer_Normalized"] <= hi)]
                    if metric == "accuracy":
                        y = float(band_df[acc_col].mean())
                    else:
                        y = float((band_df[acc_col] - band_df[ctrl_col]).mean())
                    ys.append(y)

                line_color = _shade(color, shade_strength)
                ax.plot(x, ys, color=line_color, linewidth=3.0, label=f"{family_name} ({band_label})")

                if task == "inflection" and metric == "accuracy":
                    legend_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=line_color,
                            linestyle="-",
                            linewidth=3.0,
                            label=f"{family_name} ({band_label})",
                        )
                    )

        ax.set_title(title, fontsize=20, pad=10)
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Selectivity")
        ax.tick_params(axis="both", which="major", length=8, width=1.8)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])

    fig.text(
        0.5,
        -0.06,
        "Normalized training progress (%)",
        ha="center",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=6,
            frameon=True,
            fontsize=18,
            title="Layer depth",
        )

    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"{save_name}{'_pca_' + str(pca_dim) if pca else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved training-progress depth-band figure to {out_path}")

def plot_checkpoint_training_progress_depth_3d(
    dataset: str,
    olmo_models: list[str],
    pythia_models: list[str],
    probe: str = "reg",
    output_dir: str = "figures3",
    save_name: str = "olmo_pythia_training_progress_depth_3d",
    pca: bool = False,
    pca_dim: int = 50,
    filename = None,
    elev: float = 20,
    azim: float = -70,
    alpha: float = 0.75,
):
    panels = [
        ("inflection", "accuracy"),
        ("inflection", "selectivity"),
        ("lexeme", "accuracy"),
        ("lexeme", "selectivity"),
    ]

    family_specs = [
        ("OLMo-2-7B", olmo_models, "olmo2-7b", "#1f77b4"),
        ("Pythia-6.9B", pythia_models, "pythia-6.9b", "#2ca02c"),
    ]

    def training_progress_percent(model_name: str, base_name: str, min_v: float, max_v: float) -> float:
        v = extract_checkpoint_progress(model_name, base_name)
        if v == float("inf"):
            v = max_v
        if max_v == min_v:
            return 100.0
        return 100.0 * (v - min_v) / (max_v - min_v)

    fig = plt.figure(figsize=(26, 11))
    axes = [
        [fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d") for col in range(4)]
        for row in range(2)
    ]
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.09, top=0.93, wspace=0.08, hspace=0.14)

    # Precompute Z ranges per metric to keep each column comparable across rows.
    z_ranges: dict[str, tuple[float, float]] = {}
    for _task, metric in panels:
        z_ranges[metric] = (float("inf"), float("-inf"))

    surfaces: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family_name, models, base_name, _color in family_specs:
        ordered_models = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))
        raw = [extract_checkpoint_progress(m, base_name) for m in ordered_models]
        finite = [v for v in raw if v != float("inf")]
        min_v = min(finite) if finite else 0.0
        max_v = max(finite) if finite else 1.0
        x = np.array([training_progress_percent(m, base_name, min_v, max_v) for m in ordered_models])

        for task, metric in panels:
            y_depth = None
            zs = []
            for model in ordered_models:
                probe_dir = os.path.join("..", "output", "probes", f"{dataset}_{model}_{task}_{probe}")
                if pca:
                    probe_dir += f"_pca_{pca_dim}"
                csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                df = pd.read_csv(csv_path)
                acc_col, ctrl_col = get_acc_columns(df, task)
                df = df.sort_values("Layer").reset_index(drop=True)
                df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                if y_depth is None:
                    y_depth = (df["Layer_Normalized"].to_numpy() * 100.0).astype(float)
                if metric == "accuracy":
                    z = df[acc_col].to_numpy(dtype=float)
                else:
                    z = (df[acc_col] - df[ctrl_col]).to_numpy(dtype=float)
                zs.append(z)

            if y_depth is None or not zs:
                continue

            Z = np.stack(zs, axis=1)
            X, Y = np.meshgrid(x, y_depth)
            surfaces[(family_name, task, metric)] = (X, Y, Z)

            zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            cur_min, cur_max = z_ranges[metric]
            z_ranges[metric] = (min(cur_min, zmin), max(cur_max, zmax))

    for row, (family_name, _models, _base_name, color) in enumerate(family_specs):
        for col, (task, metric) in enumerate(panels):
            ax = axes[row][col]
            key = (family_name, task, metric)
            if key in surfaces:
                X, Y, Z = surfaces[key]
                ax.plot_surface(
                    X,
                    Y,
                    Z,
                    color=color,
                    alpha=alpha,
                    linewidth=0.1,
                    antialiased=True,
                    shade=True,
                    edgecolor='none',
                    rstride=2,
                    cstride=2,
                )

            if row == 0:
                ax.set_title("Accuracy" if metric == "accuracy" else "Selectivity", pad=12, fontsize=28)
            
            if row == 1 and col in [1, 2]:
                ax.set_xlabel("Training (%)", labelpad=10, fontsize=22)
            else:
                ax.set_xlabel("")

            if col == 0:
                ax.set_ylabel("Depth (%)", labelpad=10, fontsize=22)
            else:
                ax.set_ylabel("")

            if col == 0:
                z_label = "Accuracy" if metric == "accuracy" else "Selectivity"
                ax.set_zlabel(z_label, labelpad=10, fontsize=22)
            else:
                ax.set_zlabel("")
                ax.set_zticklabels([])

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            zmin, zmax = z_ranges[metric]
            if np.isfinite(zmin) and np.isfinite(zmax) and zmin != zmax:
                ax.set_zlim(zmin, zmax)
                z_range = zmax - zmin
                if z_range > 0:
                    n_zticks = 4
                    z_ticks = np.linspace(zmin, zmax, n_zticks)
                    ax.set_zticks(z_ticks)
                    if col == 0:
                        z_labels = [f"{z:.2f}" if z_range < 0.5 else f"{z:.1f}" for z in z_ticks]
                        ax.set_zticklabels(z_labels)
            
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(axis="x", labelsize=16, pad=1, width=1.2, length=5)
            ax.tick_params(axis="y", labelsize=16, pad=1, width=1.2, length=5)
            ax.tick_params(axis="z", labelsize=16, pad=3, width=1.2, length=5)
            
            ax.grid(False)
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
            ax.xaxis.pane.set_linewidth(1.0)
            ax.yaxis.pane.set_linewidth(1.0)
            ax.zaxis.pane.set_linewidth(1.0)
            
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([0, 50, 100])
            ax.set_xticklabels(['0', '50', '100'])
            ax.set_yticklabels(['0', '50', '100'])
            
            ax.xaxis.line.set_color('black')
            ax.yaxis.line.set_color('black')
            ax.zaxis.line.set_color('black')
            ax.xaxis.line.set_linewidth(1.5)
            ax.yaxis.line.set_linewidth(1.5)
            ax.zaxis.line.set_linewidth(1.5)

    fig.text(0.27, 0.97, "Inflection", ha="center", va="center", fontsize=34)
    fig.text(0.73, 0.97, "Lexeme", ha="center", va="center", fontsize=34)

    fig.text(0.008, 0.72, "OLMo-2-7B", rotation=90, va="center", ha="center", fontsize=30)
    fig.text(0.008, 0.28, "Pythia-6.9B", rotation=90, va="center", ha="center", fontsize=30)

    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"{save_name}{'_pca_' + str(pca_dim) if pca else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight", dpi=200, pad_inches=0.2)
    print(f"Saved 3D training-progress-by-depth figure to {out_path}")

def plot_checkpoint_training_heatmap(
    dataset: str,
    olmo_models: list[str],
    pythia_models: list[str],
    probe: str = "reg",
    output_dir: str = "figures3",
    save_name: str = "olmo_pythia_training_heatmap",
    pca: bool = False,
    pca_dim: int = 50,
    filename = None,
):
    panels = [
        ("inflection", "accuracy", "Inflection\nAccuracy"),
        ("inflection", "selectivity", "Inflection\nSelectivity"),
        ("lexeme", "accuracy", "Lexeme\nAccuracy"),
        ("lexeme", "selectivity", "Lexeme\nSelectivity"),
    ]

    family_specs = [
        ("OLMo-2-7B", olmo_models, "olmo2-7b"),
        ("Pythia-6.9B", pythia_models, "pythia-6.9b"),
    ]

    def training_progress_percent(model_name: str, base_name: str, min_v: float, max_v: float) -> float:
        v = extract_checkpoint_progress(model_name, base_name)
        if v == float("inf"):
            v = max_v
        if max_v == min_v:
            return 100.0
        return 100.0 * (v - min_v) / (max_v - min_v)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = np.atleast_2d(axes)

    for row, (family_name, models, base_name) in enumerate(family_specs):
        ordered_models = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))
        raw = [extract_checkpoint_progress(m, base_name) for m in ordered_models]
        finite = [v for v in raw if v != float("inf")]
        min_v = min(finite) if finite else 0.0
        max_v = max(finite) if finite else 1.0
        x_progress = [training_progress_percent(m, base_name, min_v, max_v) for m in ordered_models]

        for col, (task, metric, title) in enumerate(panels):
            ax = axes[row, col]
            
            heatmap_data = []
            y_labels = None
            
            for model in ordered_models:
                probe_dir = os.path.join("..", "output", "probes", f"{dataset}_{model}_{task}_{probe}")
                if pca:
                    probe_dir += f"_pca_{pca_dim}"
                csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                
                if not os.path.exists(csv_path):
                    print(f"[WARN] Missing {csv_path}")
                    continue
                    
                df = pd.read_csv(csv_path)
                acc_col, ctrl_col = get_acc_columns(df, task)
                df = df.sort_values("Layer").reset_index(drop=True)
                df["Layer_Normalized"] = (df["Layer"] - df["Layer"].min()) / (df["Layer"].max() - df["Layer"].min())
                
                if y_labels is None:
                    y_labels = (df["Layer_Normalized"].to_numpy() * 100.0).astype(float)
                
                if metric == "accuracy":
                    values = df[acc_col].to_numpy(dtype=float)
                else:
                    values = (df[acc_col] - df[ctrl_col]).to_numpy(dtype=float)
                
                heatmap_data.append(values)
            
            if not heatmap_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            
            Z = np.array(heatmap_data).T
            
            if metric == "accuracy":
                vmin, vmax = 0.0, 1.0
                cmap = "YlGnBu"
            else:
                abs_max = np.abs(Z).max()
                vmin, vmax = -abs_max, abs_max
                cmap = "RdBu_r"
            
            im = ax.imshow(
                Z,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                extent=[0, 100, 0, 100],
                origin="lower",
            )
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            
            if row == 0:
                ax.set_title(title, fontsize=24, pad=10)
            
            ax.set_xlabel("Training progress (%)", fontsize=18)
            if col == 0:
                ax.set_ylabel("Layer depth (%)", fontsize=18)
            
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.tick_params(axis='both', labelsize=16)
            
            if col == 0:
                ax.text(-0.35, 0.5, family_name, transform=ax.transAxes,
                       fontsize=22, rotation=90, va='center', ha='center')

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"{save_name}{'_pca_' + str(pca_dim) if pca else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved checkpoint training heatmap to {out_path}")

def plot_checkpoint_training_delta_heatmap(
    dataset: str,
    olmo_models: list[str],
    pythia_models: list[str],
    probe: str = "reg",
    output_dir: str = "figures3",
    save_name: str = "olmo_pythia_training_delta_heatmap",
    pca: bool = False,
    pca_dim: int = 50,
    filename = None,
):
    panels = [
        ("inflection", "accuracy", "Inflection\nAccuracy Change"),
        ("inflection", "selectivity", "Inflection\nSelectivity Change"),
        ("lexeme", "accuracy", "Lexeme\nAccuracy Change"),
        ("lexeme", "selectivity", "Lexeme\nSelectivity Change"),
    ]

    family_specs = [
        ("OLMo-2-7B", olmo_models, "olmo2-7b"),
        ("Pythia-6.9B", pythia_models, "pythia-6.9b"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = np.atleast_2d(axes)

    all_acc_deltas = []
    all_sel_deltas = []

    for row, (family_name, models, base_name) in enumerate(family_specs):
        ordered_models = sorted(models, key=lambda m: extract_checkpoint_progress(m, base_name))
        first_model = ordered_models[0]
        last_model = ordered_models[-1]

        for col, (task, metric, title) in enumerate(panels):
            ax = axes[row, col]
            
            probe_dir_first = os.path.join("..", "output", "probes", f"{dataset}_{first_model}_{task}_{probe}")
            probe_dir_last = os.path.join("..", "output", "probes", f"{dataset}_{last_model}_{task}_{probe}")
            if pca:
                probe_dir_first += f"_pca_{pca_dim}"
                probe_dir_last += f"_pca_{pca_dim}"
            
            csv_first = os.path.join(probe_dir_first, f"{task}_results.csv")
            csv_last = os.path.join(probe_dir_last, f"{task}_results.csv")
            
            if not (os.path.exists(csv_first) and os.path.exists(csv_last)):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            
            df_first = pd.read_csv(csv_first)
            df_last = pd.read_csv(csv_last)
            
            acc_col, ctrl_col = get_acc_columns(df_first, task)
            
            merged = df_first.merge(df_last, on="Layer", suffixes=("_first", "_last"))
            merged = merged.sort_values("Layer").reset_index(drop=True)
            merged["Layer_Normalized"] = (merged["Layer"] - merged["Layer"].min()) / (merged["Layer"].max() - merged["Layer"].min())
            
            if metric == "accuracy":
                delta = merged[f"{acc_col}_last"] - merged[f"{acc_col}_first"]
                all_acc_deltas.extend(delta.values)
            else:
                sel_first = merged[f"{acc_col}_first"] - merged[f"{ctrl_col}_first"]
                sel_last = merged[f"{acc_col}_last"] - merged[f"{ctrl_col}_last"]
                delta = sel_last - sel_first
                all_sel_deltas.extend(delta.values)
            
            y_positions = merged["Layer_Normalized"].to_numpy() * 100
            colors = ['#2ca02c' if d >= 0 else '#d62728' for d in delta]
            
            ax.barh(y_positions, delta, height=100/len(delta), color=colors, alpha=0.8, edgecolor='none')
            ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
            
            if row == 0:
                ax.set_title(title, fontsize=24, pad=10)
            
            ax.set_xlabel("Final - first checkpoint", fontsize=18)
            if col == 0:
                ax.set_ylabel("Layer depth (%)", fontsize=18)
            
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            if col == 0:
                ax.text(-0.35, 0.5, family_name, transform=ax.transAxes,
                       fontsize=22, rotation=90, va='center', ha='center')

    for row in range(2):
        if all_acc_deltas:
            acc_max = max(abs(min(all_acc_deltas)), abs(max(all_acc_deltas)))
            for col in [0, 2]:
                axes[row, col].set_xlim(-acc_max * 1.1, acc_max * 1.1)
        
        if all_sel_deltas:
            sel_max = max(abs(min(all_sel_deltas)), abs(max(all_sel_deltas)))
            for col in [1, 3]:
                axes[row, col].set_xlim(-sel_max * 1.1, sel_max * 1.1)

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"{save_name}{'_pca_' + str(pca_dim) if pca else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved checkpoint training delta heatmap to {out_path}")

def get_acc_columns(df, prefix):
    if f"{prefix}_Accuracy" in df.columns and f"{prefix}_ControlAccuracy" in df.columns:
        return f"{prefix}_Accuracy", f"{prefix}_ControlAccuracy"
    if "Acc" in df.columns and "controlAcc" in df.columns:
        return "Acc", "controlAcc"
    for acc_col in df.columns:
        if acc_col.lower() == f"{prefix}_accuracy":
            for ctrl_col in df.columns:
                if ctrl_col.lower() == f"{prefix}_controlaccuracy":
                    return acc_col, ctrl_col
    raise ValueError("Could not find accuracy columns in DataFrame.")

def fit_and_store_regression(df, model_name, task, probe, all_results):
    for col_prefix in ["", "control_"]:
        task_col = f"{task}_{'Accuracy' if col_prefix == '' else 'ControlAccuracy'}"
        if task_col in df.columns:
            y = df[task_col].values
        else:
            y = df["Acc"].values if col_prefix == "" else df["controlAcc"].values
        X = df["Layer_Normalized"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        all_results.append({
            "model": model_name,
            "task": task,
            "probe": probe,
            "type": "linguistic" if col_prefix == "" else "control",
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "r2": r2,
        })

def get_tick_values(ymin, ymax, min_ticks=6):
    """
    Compute tick values and labels for a given y-axis range, ensuring at least min_ticks (default 6) ticks,
    including start and end. Returns (ticks, labels).
    """
    span = ymax - ymin
    if span == 0:
        return np.array([ymin]), [f"{ymin:.1f}"]
    # Try to find a "nice" step size
    raw_step = span / (min_ticks - 1)
    # Use a set of "nice" steps
    nice_steps = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0])
    step = nice_steps[np.searchsorted(nice_steps, raw_step, side="left")]
    # Compute ticks
    first_tick = np.ceil(ymin / step) * step
    last_tick = np.floor(ymax / step) * step
    ticks = np.arange(first_tick, last_tick + step/2, step)
    # Ensure start and end are included
    if abs(ticks[0] - ymin) > 1e-8:
        ticks = np.insert(ticks, 0, ymin)
    if abs(ticks[-1] - ymax) > 1e-8:
        ticks = np.append(ticks, ymax)
    # Remove duplicates and sort
    ticks = np.unique(np.round(ticks, 8))
    # Format labels
    if step < 0.1:
        labels = [f"{y:.2f}" for y in ticks]
    else:
        labels = [f"{y:.1f}" for y in ticks]
    return ticks, labels

def _get_base_name_from_models(models: list[str]) -> str | None:
    """Detect the base model name from a list of checkpoint models."""
    for m in models:
        if "pythia-6.9b" in m and "step" not in m:
            return "pythia-6.9b"
        if "olmo2-7b" in m and "stage1-step" not in m:
            return "olmo2-7b"
    # Fallback: check if all models share a common prefix
    if models and "pythia-6.9b" in models[0]:
        return "pythia-6.9b"
    if models and "olmo2-7b" in models[0]:
        return "olmo2-7b"
    return None

def plot_checkpoint_linguistic_accuracy(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    save_name="checkpoint_linguistic_accuracy",
    pca: bool = False,
    pca_dim: int = 50,
    linguistic_filename: str = None,
    ylim: tuple = ((0, 1), (0, 1)),
):
    probe_types = ["reg", "nn"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)
    all_regression_results = []

    # Sort model list by checkpoint progress (earliest first for legend ordering)
    base_name = _get_base_name_from_models(model_list)
    if base_name:
        sorted_model_list = sorted(model_list, key=lambda m: extract_checkpoint_progress(m, base_name))
    else:
        sorted_model_list = model_list

    aspect_ratio, base_height = 5.5, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * (2 * base_height + 2) * 0.75
    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.atleast_2d(axes)

    def plot_panel(fig, axes):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                probe = probe_types[col]
                ax = axes[row, col]
                if row == 0:
                    ax.set_title(f"{titles[col]}", pad=15, loc='center', fontsize=40)
                for i, model in enumerate(sorted_model_list):
                    probe_dir = os.path.join("..", "output", "probes",
                                f"{dataset}_{model}_{task}_{probe}")
                    if pca:
                        probe_dir += f"_pca_{pca_dim}"
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    if not os.path.exists(csv_path):
                        print(f"[WARN] Missing results for model: {model} at {csv_path}")
                        continue
                    df = pd.read_csv(csv_path)
                    try:
                        acc_col, ctrl_col = get_acc_columns(df, task)
                        df["Layer_Normalized"] = (
                            df["Layer"] - df["Layer"].min()
                        ) / (df["Layer"].max() - df["Layer"].min())
                        y = df[acc_col]
                        fit_and_store_regression(df, model, task, probe, all_regression_results)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=3.0,
                            color=get_model_color(model, sorted_model_list),
                        )
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=22)
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                ylabel = "Lexeme Accuracy" if row == 0 else "Inflection Accuracy"
                yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                if col == 0:
                    ax.yaxis.set_tick_params(labelleft=True)
                    ax.set_yticklabels(ylabels, fontsize=24)
                    ax.set_ylabel(ylabel, labelpad=30, fontsize=34)
                else:
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_yticklabels([])
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15, fontsize=34)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

    plot_panel(fig, axes)
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flatten()]
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    if legend_items:
        handles, labels = zip(*legend_items)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0, 0.05, 1, 0.16),
            ncol=4,
            mode="expand",
            frameon=True
        )
    fig.tight_layout(rect=[0, 0.18, 1, 1], w_pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    filename = linguistic_filename or f"checkpoint_linguistic_accuracy{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved checkpoint linguistic accuracy figure to {os.path.join(output_dir, filename)}")

    # regression_df = pd.DataFrame(all_regression_results)
    # regression_filepath = os.path.join(output_dir, "checkpoint_linguistic_accuracy_regression_results.csv")
    # regression_df.to_csv(regression_filepath, index=False)
    # print(f"Saved checkpoint linguistic accuracy regression results to {regression_filepath}")

def plot_checkpoint_selectivity(
    dataset: str,
    model_list: list[str],
    output_dir="figures3",
    save_name="checkpoint_classifier_selectivity",
    pca: bool = False,
    pca_dim: int = 50,
    selectivity_filename: str = None,
    ylim: tuple = ((-0.5, 0.5), (-0.5, 0.5)),
):
    probe_types = ["reg", "nn"]
    titles = ["Linear Regression", "MLP"]
    tasks = ["lexeme", "inflection"]
    n_rows, n_cols = len(tasks), len(probe_types)

    # Sort model list by checkpoint progress (earliest first for legend ordering)
    base_name = _get_base_name_from_models(model_list)
    if base_name:
        sorted_model_list = sorted(model_list, key=lambda m: extract_checkpoint_progress(m, base_name))
    else:
        sorted_model_list = model_list

    aspect_ratio, base_height = 5.5, 5
    fig_width = n_cols * base_height * aspect_ratio / 2.0
    fig_height = n_rows * (2 * base_height + 2) * 0.75
    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = np.atleast_2d(axes)

    def plot_panel(fig, axes):
        for row, task in enumerate(tasks):
            for col in range(n_cols):
                probe = probe_types[col]
                ax = axes[row, col]
                if row == 0:
                    ax.set_title(f"{titles[col]}", pad=15, loc='center', fontsize=40)
                for i, model in enumerate(sorted_model_list):
                    probe_dir = os.path.join("..", "output", "probes",
                                f"{dataset}_{model}_{task}_{probe}")
                    if pca:
                        probe_dir += f"_pca_{pca_dim}"
                    csv_path = os.path.join(probe_dir, f"{task}_results.csv")
                    if not os.path.exists(csv_path):
                        print(f"[WARN] Missing results for model: {model} at {csv_path}")
                        continue
                    df = pd.read_csv(csv_path)
                    try:
                        acc_col, ctrl_col = get_acc_columns(df, task)
                        df["Layer_Normalized"] = (
                            df["Layer"] - df["Layer"].min()
                        ) / (df["Layer"].max() - df["Layer"].min())
                        y = df[acc_col] - df[ctrl_col]
                        model_color = get_model_color(model, sorted_model_list)
                        ax.plot(
                            df["Layer_Normalized"], y,
                            label=model_names.get(model, model),
                            linewidth=3.0,
                            color=model_color,
                        )
                    except Exception:
                        ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=22)
                ax.tick_params(axis='both', which='major', length=10, width=2)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f"{x*100:.0f}" for x in np.arange(0, 1.1, 0.2)])
                row_ylim = ylim[row] if isinstance(ylim, (list, tuple)) and len(ylim) > row else (0, 1)
                ylabel = "Lexeme Selectivity" if row == 0 else "Inflection Selectivity"
                yticks, ylabels = get_tick_values(row_ylim[0], row_ylim[1])
                ax.set_ylim(*row_ylim)
                ax.set_yticks(yticks)
                if col == 0:
                    ax.yaxis.set_tick_params(labelleft=True)
                    ax.set_yticklabels(ylabels, fontsize=24)
                    ax.set_ylabel(ylabel, labelpad=30, fontsize=34)
                else:
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.set_yticklabels([])
                ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
                if row == 1:
                    ax.set_xlabel("Normalized layer number (%)", labelpad=15, fontsize=34)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

    plot_panel(fig, axes)
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flatten()]
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])
    seen = set()
    legend_items = []
    for h, l in zip(handles, labels):
        if l not in seen:
            legend_items.append((h, l))
            seen.add(l)
    if legend_items:
        handles, labels = zip(*legend_items)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0, 0.05, 1, 0.16),
            ncol=4,
            mode="expand",
            frameon=True
        )
    fig.tight_layout(rect=[0, 0.18, 1, 1], w_pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    filename = selectivity_filename or f"checkpoint_classifier_selectivity{'_pca_' + str(pca_dim) if pca else ''}.png"
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    print(f"Saved checkpoint classifier selectivity figure to {os.path.join(output_dir, filename)}")

# Define checkpoint model lists
olmo_models = [
    "olmo2-7b",
    "olmo2-7b_stage1-step5000-tokens21B",
    "olmo2-7b_stage1-step40000-tokens168B",
    "olmo2-7b_stage1-step97000-tokens407B",
    "olmo2-7b_stage1-step179000-tokens751B",
    "olmo2-7b_stage1-step282000-tokens1183B",
    "olmo2-7b_stage1-step409000-tokens1716B",
    "olmo2-7b_stage1-step559000-tokens2345B",
    "olmo2-7b_stage1-step734000-tokens3079B",
]

pythia_models = [
    "pythia-6.9b",
    "pythia-6.9b_step1",
    "pythia-6.9b_step64",
    "pythia-6.9b_step6000",
    "pythia-6.9b_step19000",
    "pythia-6.9b_step37000",
    "pythia-6.9b_step57000",
    "pythia-6.9b_step82000",
    "pythia-6.9b_step111000",
]

dataset = "ud_gum_dataset"

# Plot Olmo linguistic accuracy
plot_checkpoint_linguistic_accuracy(
    dataset,
    olmo_models,
    pca=False,
    linguistic_filename="olmo_linguistic_accuracy.png",
    ylim=((0, 1), (0, 1)),
)

# Plot Olmo selectivity
plot_checkpoint_selectivity(
    dataset,
    olmo_models,
    pca=False,
    selectivity_filename="olmo_classifier_selectivity.png",
    ylim=((-0.5, 0.5), (-0.5, 0.5)),
)

# Plot Pythia linguistic accuracy
plot_checkpoint_linguistic_accuracy(
    dataset,
    pythia_models,
    pca=False,
    linguistic_filename="pythia_linguistic_accuracy.png",
    ylim=((0, 1), (0, 1)),
)

# Plot Pythia selectivity
plot_checkpoint_selectivity(
    dataset,
    pythia_models,
    pca=False,
    selectivity_filename="pythia_classifier_selectivity.png",
    ylim=((-0.5, 0.5), (-0.5, 0.5)),
)

# Combined training delta plot (final - earliest) for both model families.
plot_checkpoint_training_deltas_by_depth(
    dataset,
    olmo_models,
    pythia_models,
    probe="reg",
    pca=False,
    filename="olmo_pythia_training_deltas_by_depth.png",
)

# Training progress curves aggregated over depth bands.
plot_checkpoint_vs_training_progress_depth_bands(
    dataset,
    olmo_models,
    pythia_models,
    probe="reg",
    pca=False,
    filename="olmo_pythia_training_progress_depth_bands.png",
)

# 3D: training progress × depth × metric (no depth bands).
plot_checkpoint_training_progress_depth_3d(
    dataset,
    olmo_models,
    pythia_models,
    probe="reg",
    pca=False,
    filename="olmo_pythia_training_progress_depth_3d.png",
)

# Heatmap showing training evolution across checkpoints and layers
plot_checkpoint_training_heatmap(
    dataset,
    olmo_models,
    pythia_models,
    probe="reg",
    pca=False,
    filename="olmo_pythia_training_heatmap.png",
)

# Delta heatmap showing change from first to last checkpoint
plot_checkpoint_training_delta_heatmap(
    dataset,
    olmo_models,
    pythia_models,
    probe="reg",
    pca=False,
    filename="olmo_pythia_training_delta_heatmap.png",
)
