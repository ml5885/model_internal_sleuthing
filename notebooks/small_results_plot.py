import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurable values ---
MODELS = [
    "llama3-8b", "llama3-8b-instruct", "pythia-6.9b", "pythia-6.9b-tulu"
]
MODEL_NAMES = {
    "llama3-8b": "Llama 3 8B",
    "llama3-8b-instruct": "Llama 3 8B Instruct",
    "pythia-6.9b": "Pythia 6.9B",
    "pythia-6.9b-tulu": "Pythia 6.9B Tulu",
}
DATASET = "ud_gum_dataset"
PROBE_TYPES = [("inflection", "reg"), ("lexeme", "reg"), ("inflection", "nn"), ("lexeme", "nn")]
COLORS = {"task": sns.color_palette("Set2")[0], "control": sns.color_palette("Set2")[1]}
OUTPUT_DIR = "figs"
PCA = False
PCA_DIM = 50

# METRIC = "Acc"           # e.g., "Acc", "F1", "Top5"
METRIC = "F1"
# CONTROL_METRIC = "controlAcc"  # e.g., "controlAcc", "controlF1", "controlTop5"
CONTROL_METRIC = "controlF1"
SAVE_NAME = "all_probes_" + METRIC

sns.set_style("white")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

def get_metric_cols(df):
    if METRIC in df.columns and CONTROL_METRIC in df.columns:
        return METRIC, CONTROL_METRIC
    raise ValueError(f"Columns {METRIC} and/or {CONTROL_METRIC} not found in dataframe.")

def plot_probe(ax, df, metric_col, control_col, task):
    ax.plot(df["Layer"], df[metric_col], label=f"{task} task", color=COLORS["task"], marker="o", markersize=4)
    ax.plot(df["Layer"], df[control_col], label=f"{task} control", color=COLORS["control"], marker="x", markersize=4)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{task.capitalize()} {metric_col}")

def plot_all_probes():
    nrows, ncols = len(PROBE_TYPES), len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    for col, model in enumerate(MODELS):
        for row, (task, probe_type) in enumerate(PROBE_TYPES):
            ax = axes[row, col] if nrows > 1 else axes[col]
            probe_dir = f"../output/probes/{DATASET}_{model}_{task}_{probe_type}"
            if PCA: probe_dir += f"_pca_{PCA_DIM}"
            csv_path = os.path.join(probe_dir, f"{task}_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                try:
                    metric_col, control_col = get_metric_cols(df)
                    plot_probe(ax, df, metric_col, control_col, task)
                except Exception as e:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            if row == 0:
                ax.set_title(MODEL_NAMES.get(model, model))
            if row == nrows - 1:
                ax.set_xlabel("Layer")
                
    fig.text(0.03, 0.75, "Linear probe", va="center", rotation="vertical", fontsize=22)
    fig.text(0.03, 0.29, "MLP probe", va="center", rotation="vertical", fontsize=22)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{SAVE_NAME}.png"), dpi=300)
    print(f"Saved to {os.path.join(OUTPUT_DIR, f'{SAVE_NAME}.png')}")

if __name__ == "__main__":
    plot_all_probes()
