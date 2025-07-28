import argparse
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import MODEL_DISPLAY_NAMES

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 100
plt.rcParams.update(
    {
        "font.size": 30,
        "axes.labelsize": 34,
        "axes.titlesize": 26,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
        "legend.title_fontsize": 30,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.5,
        "lines.markersize": 6,
    }
)

LINEWIDTH = 1.0

MODEL_COLORS = {
    "gpt2": "#1f77b4",
    "gpt2-large": "#ff7f0e",
    "gpt2-xl": "#2ca02c",
    "qwen2": "#d62728",
    "qwen2-instruct": "#9467bd",
    "qwen2.5-7B": "#8c564b",
    "qwen2.5-7B-instruct": "#e377c2",
    "gemma2b": "#7f7f7f",
    "gemma2b-it": "#bcbd22",
    "bert-base-uncased": "#17becf",
    "bert-large-uncased": "#aec7e8",
    "deberta-v3-large": "#ffbb78",
    "llama3-8b": "#98df8a",
    "llama3-8b-instruct": "#c5b0d5",
    "pythia-6.9b": "#ff9896",
    "pythia-6.9b-tulu": "#c49c94",
    "olmo2-7b-instruct": "#f7b6d2",
    "olmo2-7b": "#dbdb8d",
}

LAMBDA_STYLES = {
    1: dict(marker="o"),
    5: dict(marker="s"),
    10: dict(marker="^"),
    20: dict(marker="D"),
    100: dict(marker="*"),
}

def collect_all_results(steering_dir: str, models: list[str], dataset: str) -> pd.DataFrame:
    rows = []
    for model in models:
        pat = re.compile(
            rf"^{re.escape(dataset)}_{re.escape(model)}_(?P<probe>\w+)_lambda(?P<lam>\d+\.?\d*)$"
        )
        for d in os.listdir(steering_dir):
            m = pat.match(d)
            if not m:
                continue
            probe_type = m.group("probe")
            lam = float(m.group("lam"))
            csv = os.path.join(steering_dir, d, "steering_summary.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            df["model"] = model
            df["probe_type"] = probe_type
            df["lambda"] = lam
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["layer"] = out["layer"].astype(int)

    # Normalize layer number per model
    max_layers = out.groupby("model")["layer"].transform("max")
    out["normalized_layer"] = 100 * out["layer"] / max_layers

    return out


def lambda_legend_handles(lam_values):
    handles = []
    labels = []
    for lam in sorted(lam_values):
        style = LAMBDA_STYLES.get(lam, dict(marker="o"))
        h = plt.Line2D(
            [], [],
            color="black",
            marker=style["marker"],
            linestyle="none",
            markersize=6,
        )
        handles.append(h)
        labels.append(f"lambda {lam:g}")
    return handles, labels

def model_legend_handles(model_values):
    handles = []
    labels = []
    sorted_models = sorted(model_values, key=lambda m: list(MODEL_COLORS.keys()).index(m) if m in MODEL_COLORS else m)
    for model in sorted_models:
        h = plt.Line2D(
            [], [],
            color=MODEL_COLORS.get(model, "gray"),
            marker="s",
            linestyle="-",
            linewidth=LINEWIDTH,
        )
        handles.append(h)
        labels.append(MODEL_DISPLAY_NAMES.get(model, model))
    return handles, labels

def adjust_color_brightness(color, factor):
    rgb = mcolors.to_rgb(color)
    return tuple(min(1, max(0, c * factor)) for c in rgb)

def plot_each_model_separately(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    if df.empty:
        return

    figsize = (18, 12)
    grid = dict(linestyle="--", alpha=0.4, linewidth=0.8)
    metrics = {
        "mean_prob_change": "Mean Probability Change",
        "flip_rate": "Prediction Flip Rate",
    }

    for metric in metrics:
        os.makedirs(os.path.join(out_dir, metric), exist_ok=True)

    model_values = sorted(df["model"].unique())

    all_lam_values = sorted(df["lambda"].unique())
    n_lam = len(all_lam_values)
    viridis_cmap = cm.get_cmap("viridis", n_lam)
    lam_to_color = {lam: viridis_cmap(i) for i, lam in enumerate(all_lam_values)}

    for model in model_values:
        model_df = df[df["model"] == model]
        probe_values = sorted(model_df["probe_type"].unique())
        lam_values = sorted(model_df["lambda"].unique())

        for probe in probe_values:
            sub_probe = model_df[model_df["probe_type"] == probe]
            for metric, ylabel in metrics.items():
                fig, ax = plt.subplots(figsize=figsize)

                for lam in lam_values:
                    g = sub_probe[sub_probe["lambda"] == lam]
                    if not g.empty:
                        color = lam_to_color[lam]
                        ax.plot(
                            g["normalized_layer"],
                            g[metric],
                            color=color,
                            marker="o",
                            markersize=6,
                            linestyle="-",
                            linewidth=LINEWIDTH,
                            label=f"λ={lam:g}",
                        )

                ax.set_title(f"{ylabel} - {probe.upper()} - {MODEL_DISPLAY_NAMES.get(model, model)}")
                ax.set_xlabel("Normalized Layer Number (%)")
                ax.set_ylabel(ylabel)
                ax.grid(**grid)
                ax.legend(title="Lambda", fontsize=16, title_fontsize=18, loc="best")
                fig.tight_layout()
                fname = f"{metric}/{dataset}_{model}_{probe}_{metric}.png"
                fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
                print(f"Saved plot: {fname}")
                plt.close(fig)

def plot_all_models_one_figure(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    if df.empty:
        return
    figsize = (18, 12)
    grid = dict(linestyle="--", alpha=0.4, linewidth=0.8)
    metrics = {
        "mean_prob_change": "Mean Probability Change",
        "flip_rate": "Prediction Flip Rate",
    }
    for probe in df["probe_type"].unique():
        sub_probe = df[df["probe_type"] == probe]
        model_values = sorted(sub_probe["model"].unique())
        lam_values = sorted(sub_probe["lambda"].unique())
        for metric, ylabel in metrics.items():
            fig, ax = plt.subplots(figsize=figsize)
            for model in model_values:
                for lam in lam_values:
                    g = sub_probe[(sub_probe["model"] == model) & (sub_probe["lambda"] == lam)]
                    if not g.empty:
                        ax.plot(
                            g["normalized_layer"],
                            g[metric],
                            color=MODEL_COLORS.get(model, "gray"),
                            marker=LAMBDA_STYLES.get(lam, dict(marker="o"))["marker"],
                            linestyle="-",
                            linewidth=LINEWIDTH,
                        )
            ax.set_title(f"{ylabel} - {probe.upper()} - {dataset}")
            ax.set_xlabel("Normalized Layer Number (%)")
            ax.set_ylabel(ylabel)
            ax.grid(**grid)

            model_legend_h, model_legend_l = model_legend_handles(model_values)
            model_legend = fig.legend(
                model_legend_h,
                model_legend_l,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.25),
                ncol=4,
                frameon=True,
            )
            fig.add_artist(model_legend)

            lambda_legend_h, lambda_legend_l = lambda_legend_handles(lam_values)
            fig.legend(
                lambda_legend_h,
                lambda_legend_l,
                title="Lambda",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.33),
                ncol=5,
                frameon=True,
            )
            fig.tight_layout(rect=[0, 0.3, 1, 1])
            fname = f"{dataset}_{probe}_{metric}_ALL_MODELS.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            print(f"Saved plot: {fname}")
            plt.close(fig)

def plot_each_lambda_separately(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    if df.empty:
        return
    figsize = (29, 16)
    grid = dict(linestyle="--", alpha=0.4, linewidth=0.8)
    metrics = {
        "mean_prob_change": "Mean Probability Change",
        "flip_rate": "Prediction Flip Rate",
    }
    lam_values = sorted(df["lambda"].unique())
    n_lam = len(lam_values)
    n_plots = min(n_lam, 5)
    ncols = 3
    nrows = 2
    for probe in sorted(df["probe_type"].unique()):
        sub_probe = df[df["probe_type"] == probe]
        model_values = sorted(sub_probe["model"].unique())
        for metric, ylabel in metrics.items():
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = axes.flatten()
            for idx, lam in enumerate(lam_values[:5]):
                ax = axes[idx]
                sub_lam = sub_probe[sub_probe["lambda"] == lam]
                for model in model_values:
                    g = sub_lam[sub_lam["model"] == model]
                    if not g.empty:
                        ax.plot(
                            g["normalized_layer"],
                            g[metric],
                            color=MODEL_COLORS.get(model, "gray"),
                            marker="o",
                            linestyle="-",
                            linewidth=LINEWIDTH,
                            label=MODEL_DISPLAY_NAMES.get(model, model),
                        )
                ax.set_title(r"$\lambda$={:g}".format(lam), fontsize=36)
                if idx // ncols == nrows - 1 or idx % ncols == ncols - 1:
                    ax.set_xlabel("Normalized Layer Number (%)")
                else:
                    ax.set_xlabel("")
                if idx % ncols == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel("")
                ax.grid(**grid)
            for idx in range(n_lam, 5):
                axes[idx].axis("off")
            legend_ax = axes[5]
            legend_ax.axis("off")
            handles, labels = [], []
            for model in model_values:
                handles.append(
                    plt.Line2D(
                        [], [],
                        color=MODEL_COLORS.get(model, "gray"),
                        marker="o",
                        linestyle="-",
                        linewidth=LINEWIDTH,
                        label=MODEL_DISPLAY_NAMES.get(model, model),
                    )
                )
                labels.append(MODEL_DISPLAY_NAMES.get(model, model))
            legend_ax.legend(
                handles, labels, title="Model", fontsize=24, title_fontsize=26, loc="center",
                ncol=2, borderaxespad=0.2, handletextpad=0.5, columnspacing=1.2
            )
            fig.subplots_adjust(top=0.88)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fname = f"{dataset}_{probe}_{metric}_LAMBDAS_MULTIPLOT.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
            print(f"Saved plot: {fname}")
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_dir", required=True)
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = collect_all_results(args.steering_dir, args.models, args.dataset)
    if df.empty:
        print("No steering results found.")
        return

    plot_all_models_one_figure(df, args.dataset, args.output_dir)
    # plot_each_model_separately(df, args.dataset, args.output_dir)
    plot_each_lambda_separately(df, args.dataset, args.output_dir)

if __name__ == "__main__":
    main()