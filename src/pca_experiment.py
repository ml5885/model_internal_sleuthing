import argparse
import os
import re
import math
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# set global plotting style
sns.set_style('white')
mpl.rcParams.update({
    'figure.dpi': 150,
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import list_shards, shard_loader
from src import config

model_names = config.MODEL_DISPLAY_NAMES
# Mapping from table names to model keys
# Use config.MODEL_TABLE_MAPPING

table_model_mapping = config.MODEL_TABLE_MAPPING

def load_layer_activations(activations_dir, layer_idx):
    shards = list_shards(activations_dir)
    arrays = [shard_loader(path, layer_idx) for path in shards]
    return np.vstack(arrays)

def compute_intrinsic_dims(X, thresholds):
    """
    Computes intrinsic dimensions for given thresholds, returns dims dict.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=X.shape[1], random_state=config.SEED)
    pca.fit(Xc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims = {th: int(np.searchsorted(cumvar, th) + 1) for th in thresholds}
    dims[1.0] = X.shape[1]
    return dims

def single_model_analysis(model, dataset, thresholds, max_layers, out_dir, reuse_existing=False):
    display = model_names.get(model, model)
    activ_dir = os.path.join(config.OUTPUT_DIR, f'{model}_{dataset}_reps')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'intrinsic_dims_by_layer.csv')
    if reuse_existing and os.path.exists(csv_path):
        print(f'Found existing CSV file for {model}.')
        df = pd.read_csv(csv_path)
    else:
        shards = list_shards(activ_dir)
        if not shards:
            raise RuntimeError(f'No shards found in {activ_dir}')
        sample = np.load(shards[0], mmap_mode='r')['activations']
        _, n_layers, _ = sample.shape
        if max_layers is not None:
            n_layers = min(n_layers, max_layers)

        records = []
        for layer in tqdm(range(n_layers), desc=display):
            X = load_layer_activations(activ_dir, layer)
            dims = compute_intrinsic_dims(X, thresholds)
            rec = {'layer': layer}
            rec.update({f'dim_{int(th*100)}': n for th, n in dims.items()})
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        df.to_csv(csv_path, index=False)

    return df

def extract_layer_values(df, column):
    """Return first / mid / final layer values for a column, or '--' if absent."""
    if df is None or df.empty:
        return "--", "--", "--"
    n_layers = len(df)
    first = df.iloc[0][column]
    mid   = df.iloc[n_layers // 2][column]
    final = df.iloc[-1][column]
    return str(first), str(mid), str(final)

def generate_latex_table(dfs_dict, table_model_mapping, out_dir):
    latex = r"""\begin{table*}[t]
\centering
\small
\renewcommand\arraystretch{1.3}
\resizebox{\linewidth}{!}{%
  \begin{tabular}{@{}l c ccc ccc ccc@{}}
    \toprule
    \multirow{2}{*}{Model} & \multirow{2}{*}{$d_\text{model}$} &
      \multicolumn{3}{c}{ID$_{50}$} &
      \multicolumn{3}{c}{ID$_{70}$} &
      \multicolumn{3}{c}{ID$_{90}$} \\
    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
      & & First & Mid & Final & First & Mid & Final & First & Mid & Final \\
    \midrule
"""
    for display_name, model_key in table_model_mapping.items():
        df = dfs_dict.get(model_key)
        id50_first, id50_mid, id50_final = extract_layer_values(df, "dim_50")
        id70_first, id70_mid, id70_final = extract_layer_values(df, "dim_70")
        id90_first, id90_mid, id90_final = extract_layer_values(df, "dim_90")

        m = re.match(r"(.+?)\s*\((\d+)\)", display_name)
        if m:
            model_disp = m.group(1).strip()
            d_model = m.group(2)
        else:
            model_disp = display_name
            d_model = "--"

        latex += (
            f"    {model_disp:<35} & {d_model} & "
            f"{id50_first} & {id50_mid} & {id50_final} & "
            f"{id70_first} & {id70_mid} & {id70_final} & "
            f"{id90_first} & {id90_mid} & {id90_final} \\\\\n"
        )

    latex += r"""    \bottomrule
  \end{tabular}}%
\caption{Number of principal-component axes required to reach 50\% (ID$_{50}$), 70\% (ID$_{70}$) and 90\% (ID$_{90}$) explained variance in the first, middle and last layers of each model.}
\label{fig:intrinsic_dim_table}
\end{table*}
"""
    out_path = os.path.join(out_dir, "intrinsic_dimensions_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {out_path}")

MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#a55194", "#393b79",
    "#637939", "#e6550d", "#9c9ede", "#f7b6d2"
]

def get_model_color(model, models):
    idx = models.index(model)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]

def plot_components_by_threshold_multiplot(dfs, models, thresholds, out_base, normalize=True):
    """
    One subplot per variance threshold; 2 rows x 4 columns, full width figure.
    Larger tick marks, 4 x-axis labels, and custom x-axis label.
    """
    steps = thresholds + [1.0]
    pct_vals = [int(th * 100) for th in steps]
    n = len(pct_vals)
    nrows, ncols = 2, 4

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(24, 8),
        sharex=True, sharey=True,
        constrained_layout={'hspace': 0.1, 'wspace': 0.05}
    )
    axs = axs.flatten()

    for i, pct in enumerate(pct_vals):
        ax = axs[i]
        for df, model in zip(dfs, models):
            disp = model_names.get(model, model)
            x = df['layer'] / df['layer'].max() * 100
            y = df[f'dim_{pct}'].astype(float)
            if normalize:
                y /= df['dim_100'].iloc[0]
            color = get_model_color(model, models)
            ax.plot(x, y, linewidth=2, label=disp, color=color)

        row = i // ncols
        col = i % ncols

        ax.set_title(f'{pct}% explained variance', fontsize=24)
        ax.set_xlim(0, 100)
        if normalize:
            ax.set_ylim(0, 1)
        ax.set_xticks([0, 33, 66, 100])
        ax.set_xticklabels(['0', '33', '66', '100'], fontsize=20)
        ax.set_yticks(np.linspace(0, 1, 5) if normalize else ax.get_yticks())
        ax.tick_params(axis='both', which='major', length=8, width=2, labelsize=20)
        ax.grid(True, linestyle=':', linewidth=1.2)

        if col != 0:
            ax.yaxis.set_tick_params(labelleft=False)
        if row != nrows - 1:
            ax.xaxis.set_tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Normalized layer number (%)', fontsize=24)

    for j in range(n, len(axs)):
        axs[j].axis('off')

    fig.supylabel('Components (fraction of max)', fontsize=28, x=-0.03)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=min(4, len(models)),
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=22
    )

    out_path = os.path.join(out_base, 'multi_components_by_thresholds.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined components-by-thresholds plot to {out_path}')

def plot_variance_by_model_multiplot(dfs, models, thresholds, out_base):
    """
    One subplot per model; curves colored by layer; shared colorbar.
    """
    steps = thresholds + [1.0]
    pct_vals = [int(th * 100) for th in steps]

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    n = len(models)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharey=True,
        constrained_layout={'hspace': 0.1}
    )
    axs = axs.flatten()

    for i, (df, model) in enumerate(zip(dfs, models)):
        ax = axs[i]
        disp = model_names.get(model, model)
        max_layer = df['layer'].max()
        max_components = df['dim_100'].iloc[0]
        for _, row in df.iterrows():
            layer = int(row['layer'])
            x = [row[f'dim_{p}'] / max_components for p in pct_vals]
            y = pct_vals
            ax.plot(x, y, color=cmap(norm(layer / max_layer)), linewidth=1.5)

        row = i // ncols
        col = i % ncols

        ax.set_title(disp, fontsize=30)
        ax.set_xlim(0, 1)
        ax.set_yscale('linear')  # linear explained variance
        ax.grid(True, linestyle=':', linewidth=0.8)

        if col != 0:
            ax.yaxis.set_tick_params(labelleft=False)
        else:
            ax.set_ylabel('Explained variance (%)', fontsize=24)

        if row != nrows - 1:
            ax.xaxis.set_tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Components (fraction of max)', fontsize=24)

    for j in range(n, len(axs)):
        axs[j].axis('off')

    cbar = fig.colorbar(
        sm,
        ax=axs.tolist(),
        orientation='vertical',
        fraction=0.02,
        pad=0.04,
        label='Layer (%)'
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels([f'{int(t*100)}%' for t in np.linspace(0, 1, 5)])

    out_path = os.path.join(out_base, 'multi_variance_by_model.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved combined variance-by-model plot to {out_path}')

def multi_model_plot(dfs, models, thresholds, max_layers, out_base):
    plot_components_by_threshold_multiplot(dfs, models, thresholds, out_base)
    plot_variance_by_model_multiplot(dfs, models, thresholds, out_base)

def main():
    parser = argparse.ArgumentParser(description='Intrinsic PCA dim per model')
    parser.add_argument('--models', '-M', nargs='+', required=True)
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--output-dir', '-o', required=True)
    parser.add_argument(
        '--thresholds', '-t', type=int, nargs='+',
        default=[50, 60, 70, 80, 90, 95, 99]
    )
    parser.add_argument('--max-layers', type=int, default=None)
    parser.add_argument('--override', '-O', action='store_true')
    args = parser.parse_args()

    thresholds = [t / 100.0 for t in args.thresholds]
    dfs, used = [], []
    dfs_dict = {}
    for model in args.models:
        out_model = os.path.join(args.output_dir, model)
        os.makedirs(out_model, exist_ok=True)
        try:
            df = single_model_analysis(
                model, args.dataset, thresholds,
                args.max_layers, out_model,
                reuse_existing=not args.override
            )
            dfs.append(df)
            used.append(model)
            dfs_dict[model] = df
        except RuntimeError as e:
            print(f"Warning: Could not process model {model}: {e}")
            dfs_dict[model] = None

    if len(used) > 1:
        multi_model_plot(dfs, used, thresholds, args.max_layers, args.output_dir)
    
    generate_latex_table(dfs_dict, table_model_mapping, args.output_dir)

if __name__ == '__main__':
    main()