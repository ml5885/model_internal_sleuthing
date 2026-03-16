import argparse
import os
import re
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from tqdm import tqdm
from sklearn.decomposition import PCA

# set global plotting style
sns.set_style('white')
mpl.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 100,
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 30,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'legend.title_fontsize': 24,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import list_shards, shard_loader
from src import config

model_names = config.MODEL_DISPLAY_NAMES
table_model_mapping = config.MODEL_TABLE_MAPPING

def load_layer_activations(activations_dir, layer_idx):
    shards = list_shards(activations_dir)
    if not shards:
        raise FileNotFoundError(f"No shards found in {activations_dir}")
    arrays = [shard_loader(path, layer_idx) for path in shards]
    return np.vstack(arrays)

def compute_intrinsic_dims(X, thresholds):
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=X.shape[1], random_state=config.SEED)
    pca.fit(Xc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims = {th: int(np.searchsorted(cumvar, th) + 1) for th in thresholds}
    dims[1.0] = X.shape[1]
    return dims

def get_activations_dir(model, dataset):
    activ_dir = os.path.join(config.OUTPUT_DIR, f'{model}_{dataset}_reps')
    
    if not os.path.exists(activ_dir):
        candidate_suffixes = ["_pos_reps", "_dep_reps", "_ner_reps", "_constituents_reps"]
        found = False
        
        def check_paths(base_dataset_name):
            for suffix in candidate_suffixes:
                alt_path = os.path.join(config.OUTPUT_DIR, f'{model}_{base_dataset_name}{suffix}')
                if os.path.exists(alt_path):
                    return alt_path
            return None

        # 1. Try with original dataset name + suffixes
        path = check_paths(dataset)
        if path:
            return path
            
        # 2. Try removing "_dataset" from dataset name if present
        if dataset.endswith("_dataset"):
            short_dataset = dataset.replace("_dataset", "")
            path = check_paths(short_dataset)
            if path:
                return path
        
        # 3. Try in "activations" subdirectory
        alt_dir = os.path.join(config.OUTPUT_DIR, "activations", f'{model}_{dataset}_reps')
        if os.path.exists(alt_dir):
            return alt_dir
            
    return activ_dir

def single_model_analysis(model, dataset, thresholds, max_layers, out_dir, reuse_existing=False):
    display = model_names.get(model, model)
    activ_dir = get_activations_dir(model, dataset)
    
    if "activations" in activ_dir or "reps" in activ_dir:
        if os.path.exists(activ_dir):
            print(f"Using activations from: {activ_dir}")

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

def sanity_check_model(model, dataset, out_dir):
    display = model_names.get(model, model)
    activ_dir = get_activations_dir(model, dataset)
    
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'sanity_check_stats.csv')
    
    try:
        shards = list_shards(activ_dir)
    except FileNotFoundError:
        print(f"Skipping {model}: No activations found at {activ_dir}")
        return None

    if not shards:
        print(f"Skipping {model}: No shards in {activ_dir}")
        return None

    sample = np.load(shards[0], mmap_mode='r')['activations']
    n_layers = sample.shape[1]
    
    records = []
    print(f"Running sanity check for {model}...")
    
    for layer in tqdm(range(n_layers), desc=f"Sanity Check {display}"):
        X = load_layer_activations(activ_dir, layer)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        abs_X = np.abs(X)
        layer_global_max = np.max(abs_X)
        median_abs_mean = np.median(np.abs(means))
        median_std = np.median(stds)
        if median_abs_mean > 1e-6:
            outlier_mean_count = np.sum(np.abs(means) > 10 * median_abs_mean)
        else:
            outlier_mean_count = 0
        if median_std > 1e-6:
            outlier_std_count = np.sum(stds > 10 * median_std)
        else:
            outlier_std_count = 0
        idx_max_mean = np.argmax(np.abs(means))
        val_max_mean = means[idx_max_mean]
        idx_max_std = np.argmax(stds)
        val_max_std = stds[idx_max_std]
        records.append({
            'layer': layer,
            'global_max_abs': layer_global_max,
            'median_abs_mean': median_abs_mean,
            'max_abs_mean': np.abs(val_max_mean),
            'idx_max_abs_mean': idx_max_mean,
            'median_std': median_std,
            'max_std': val_max_std,
            'idx_max_std': idx_max_std,
            'cnt_outlier_mean': outlier_mean_count,
            'cnt_outlier_std': outlier_std_count
        })
        
    df = pd.DataFrame(records)
    df.to_csv(report_path, index=False)
    print(f"Saved sanity check report to {report_path}")
    
    plot_sanity_stats(df, model, out_dir)
    return df

def plot_sanity_stats(df, model, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 20, 'axes.titlesize': 20})
    display = model_names.get(model, model)
    axes[0].plot(df['layer'], df['global_max_abs'], marker='o')
    axes[0].set_title(f'{display}\nMax Absolute Activation')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Max Value')
    axes[0].grid(True)
    axes[1].plot(df['layer'], df['max_abs_mean'], marker='o', color='orange')
    axes[1].set_title(f'{display}\nMax Mean (Abs) per Dim')
    axes[1].set_xlabel('Layer')
    axes[1].grid(True)
    axes[2].plot(df['layer'], df['max_std'], marker='o', color='green')
    axes[2].set_title(f'{display}\nMax Std per Dim')
    axes[2].set_xlabel('Layer')
    axes[2].grid(True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'sanity_check_plot.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sanity check plot to {out_path}")

def collect_activations_3d(model, dataset, out_dir, override=False):
    out_path = os.path.join(out_dir, 'activation_stats_3d.npy')
    if not override and os.path.exists(out_path):
        print(f"Found existing 3D stats for {model}, skipping collection.")
        return np.load(out_path)

    display = model_names.get(model, model)
    activ_dir = get_activations_dir(model, dataset)
    
    if not os.path.exists(activ_dir):
        print(f"Skipping {model}: Activations dir not found: {activ_dir}")
        return None

    try:
        shards = list_shards(activ_dir)
    except FileNotFoundError:
        print(f"Skipping {model}: No shards in {activ_dir}")
        return None
        
    if not shards:
        print(f"Skipping {model}: No shards in {activ_dir}")
        return None

    sample = np.load(shards[0], mmap_mode='r')['activations']
    # sample shape is expected to be (batch, n_layers, d_model)
    print(f"[DEBUG][{model}] sample activations shape: {sample.shape}")
    n_layers = sample.shape[1]
    d_model = sample.shape[2]
    
    print(f"Collecting 3D stats for {model} ({n_layers} layers, {d_model} dims)...")
    
    layer_stats = []
    for layer in tqdm(range(n_layers), desc=f"3D Stats {display}"):
        X = load_layer_activations(activ_dir, layer)  # (N, d_model)
        mean_abs = np.mean(np.abs(X), axis=0)         # (d_model,)

        # Per-layer debug for a few representative layers
        if layer in (0, n_layers // 2, n_layers - 1):
            print(
                f"[DEBUG][{model}] layer {layer}: "
                f"mean_abs min={mean_abs.min():.4g}, "
                f"median={np.median(mean_abs):.4g}, "
                f"max={mean_abs.max():.4g}"
            )

        layer_stats.append(mean_abs)
        
    data = np.vstack(layer_stats)  # (n_layers, d_model)

    # Global debug statistics
    print(f"[DEBUG][{model}] stacked data shape: {data.shape}")
    print(
        f"[DEBUG][{model}] data min={data.min():.4g}, "
        f"median={np.median(data):.4g}, "
        f"mean={data.mean():.4g}, "
        f"max={data.max():.4g}"
    )
    for q in (0.9, 0.95, 0.99):
        print(f"[DEBUG][{model}] data {int(q*100)}th percentile={np.quantile(data, q):.4g}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, data)
    print(f"Saved 3D stats to {out_path}")
    return data


def plot_3d_activations(models, out_dir):
    target_order = ['gpt2-large', 'pythia-6.9b', 'bert-large-uncased', 'llama3-8b']
    plot_models = []
    for m in target_order:
        if m in models:
            plot_models.append(m)
    for m in models:
        if m not in plot_models:
            plot_models.append(m)
    
    if not plot_models:
        print("No models to plot.")
        return

    fig = plt.figure(figsize=(32, 8))
    n_plots = len(plot_models)
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 22})
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95)

    for i, model in enumerate(plot_models):
        ax = fig.add_subplot(1, n_plots, i + 1, projection='3d')
        
        model_dir = os.path.join(out_dir, model)
        stats_path = os.path.join(model_dir, 'activation_stats_3d.npy')
        
        if not os.path.exists(stats_path):
            print(f"Stats not found for {model} at {stats_path}")
            continue
            
        data = np.load(stats_path)
        n_layers, d_model = data.shape

        print(f"[DEBUG][{model}] data.shape = {data.shape}")
        print(
            f"[DEBUG][{model}] data min={data.min():.4g}, "
            f"median={np.median(data):.4g}, "
            f"mean={data.mean():.4g}, "
            f"max={data.max():.4g}"
        )

        max_per_dim = data.max(axis=0)
        
        top_k = 50
        top_indices = np.argsort(max_per_dim)[-top_k:]
        top_indices = np.sort(top_indices)
        
        # For labeling, select the top 3 dims with largest max magnitude.
        label_count = 3
        label_dims_unsorted = np.argsort(max_per_dim)[-label_count:]
        # Sort by descending magnitude for better label order
        label_dims_sorted = label_dims_unsorted[np.argsort(max_per_dim[label_dims_unsorted])[::-1]]
        print(f"[DEBUG][{model}] label dims by largest max magnitude: {label_dims_sorted.tolist()}")
        print(f"[DEBUG][{model}] max_per_dim[label_dims] = {[float(v) for v in max_per_dim[label_dims_sorted]]}")

        _x = np.arange(d_model)
        _y = np.arange(n_layers)
        
        all_x = []
        all_y = []
        all_z = []
        all_dx = []
        all_dy = []
        all_dz = []
        
        for dim_idx in top_indices:
            layers_range = np.arange(n_layers)
            vals = data[:, dim_idx]
            
            all_x.append(np.full(n_layers, dim_idx))
            all_y.append(layers_range)
            all_z.append(np.zeros(n_layers))
            all_dx.append(0.8 * np.ones(n_layers))
            all_dy.append(0.6 * np.ones(n_layers))
            all_dz.append(vals)
            
        final_x = np.concatenate(all_x)
        final_y = np.concatenate(all_y)
        final_z = np.concatenate(all_z)
        final_dx = np.concatenate(all_dx)
        final_dy = np.concatenate(all_dy)
        final_dz = np.concatenate(all_dz)
        
        ax.bar3d(
            final_x, final_y, final_z,
            final_dx, final_dy, final_dz,
            shade=True, color='#1f77b4', edgecolor='#1f77b4', linewidth=0.1
        )
        
        display = model_names.get(model, model)
        ax.set_title(display, fontsize=30, pad=0)
        ax.set_xlabel('Dimension', labelpad=20, fontsize=20)
        ax.set_ylabel('Layer', labelpad=15, fontsize=20)
        ax.set_zlabel('Mean Abs Activation', labelpad=20, fontsize=20)
        
        ax.set_xticks(label_dims_sorted)
        ax.set_xticklabels([str(idx) for idx in label_dims_sorted], rotation=0)
        
        ax.tick_params(axis='z', pad=10)

        if n_layers > 10:
            ax.set_yticks(np.arange(0, n_layers, 10))
        else:
            ax.set_yticks(np.arange(n_layers))

        ax.view_init(elev=30, azim=-60)
        ax.grid(True)

    out_path = os.path.join(out_dir, 'combined_3d_activations.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved combined 3D plot to {out_path}")


def compute_layer_box_stats(activ_dir, n_layers, model_display, max_layers=None):
    stats_list = []
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
        
    for layer in tqdm(range(n_layers), desc=f"Box Stats {model_display}"):
        X = load_layer_activations(activ_dir, layer)
        vals = X.flatten()
        
        vals_min = vals.min()
        vals_max = vals.max()
        
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        
        whis_lo = q1 - 1.5 * iqr
        whis_hi = q3 + 1.5 * iqr
        
        # Clamp whiskers to min/max
        whis_lo = max(whis_lo, vals_min)
        whis_hi = min(whis_hi, vals_max)
        
        # Identify fliers
        fliers = vals[(vals < whis_lo) | (vals > whis_hi)]
        
        # Downsample fliers if too many for plotting performance
        if len(fliers) > 2000:
             # Keep top 1000 and bottom 1000 outliers to capture spikes
             fliers_sorted = np.sort(fliers)
             fliers_to_keep = np.concatenate([fliers_sorted[:1000], fliers_sorted[-1000:]])
        else:
            fliers_to_keep = fliers
            
        stats = {
            'label': str(layer),
            'mean': np.mean(vals),
            'med': med,
            'q1': q1,
            'q3': q3,
            'whislo': whis_lo,
            'whishi': whis_hi,
            'fliers': fliers_to_keep
        }
        stats_list.append(stats)
    return stats_list

def plot_box_whiskers(models, dataset, out_dir, override=False, max_layers=None):
    print("Generating box-and-whisker plots...")
    os.makedirs(out_dir, exist_ok=True)
    
    for model in models:
        display = model_names.get(model, model)
        model_out_dir = os.path.join(out_dir, model)
        os.makedirs(model_out_dir, exist_ok=True)
        stats_path = os.path.join(model_out_dir, 'layer_box_stats.npy')
        
        stats = None
        if not override and os.path.exists(stats_path):
            print(f"Loading existing box stats for {model}...")
            try:
                stats = np.load(stats_path, allow_pickle=True).tolist()
            except Exception as e:
                print(f"Failed to load stats for {model}: {e}")
                stats = None

        if stats is None:
            activ_dir = get_activations_dir(model, dataset)
            if not os.path.exists(activ_dir):
                print(f"Skipping {model}: Activations not found at {activ_dir}")
                continue
                
            try:
                shards = list_shards(activ_dir)
            except FileNotFoundError:
                print(f"Skipping {model}: No shards found.")
                continue
                
            if not shards:
                 print(f"Skipping {model}: No shards found.")
                 continue
                 
            try:
                sample = np.load(shards[0], mmap_mode='r')['activations']
                n_layers = sample.shape[1]
            except Exception as e:
                print(f"Error loading sample for {model}: {e}")
                continue
            
            print(f"Computing box stats for {model}...")
            stats = compute_layer_box_stats(activ_dir, n_layers, display, max_layers)
            np.save(stats_path, np.array(stats, dtype=object))
            print(f"Saved stats to {stats_path}")
            
        if stats:
            fig, ax = plt.subplots(figsize=(20, 10))
            
            ax.bxp(stats, showfliers=True, flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.5, 'color': 'red'})
            
            ax.set_title(f'{display} Layer Activations Distribution')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Activation Value')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            n = len(stats)
            if n > 20:
                ax.set_xticks(np.arange(0, n + 1, 5))
                ticks = np.arange(1, n + 1, 5)
                ax.set_xticks(ticks)
                ax.set_xticklabels([stats[i-1]['label'] for i in ticks])
            
            plot_path = os.path.join(model_out_dir, 'layer_box_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved box plot to {plot_path}")


def extract_layer_values(df, column):
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
    One subplot per variance threshold; single row, reduced width per subplot.
    Excludes 60% and 80% thresholds.
    """
    skip_pcts = {60, 80, 99, 100}
    steps = thresholds + [1.0]
    pct_vals = [int(th * 100) for th in steps]
    pct_vals = [p for p in pct_vals if p not in skip_pcts]
    n = len(pct_vals)
    nrows, ncols = 1, n

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 5),
        sharex=True, sharey=True,
        constrained_layout={'wspace': 0.05}
    )
    if ncols == 1:
        axs = [axs]
    axs = np.asarray(axs).flatten()

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

        ax.set_title(f'{pct}% explained variance', fontsize=20, pad=4)
        ax.set_xlim(0, 100)
        if normalize:
            ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=20)
        yticks = np.linspace(0, 1, 5) if normalize else ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{y:.1f}' if normalize else str(y) for y in yticks], fontsize=20)
        ax.tick_params(axis='both', which='major', length=8, width=2)
        ax.grid(True, linestyle=':', linewidth=1.2)

        if i != 0:
            ax.yaxis.set_tick_params(labelleft=False)

    fig.supxlabel('Normalized layer number (%)', fontsize=22, y=-0.08)
    fig.supylabel('Components (fraction of max)', x=-0.03, fontsize=19)

    handles_labels = [ax.get_legend_handles_labels() for ax in axs if hasattr(ax, "get_legend_handles_labels")]
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
            loc='lower center',
            bbox_to_anchor=(0.0, -0.7, 1.0, 0.1),
            ncol=4,
            mode="expand",
            frameon=True,
            fontsize=20,
            handletextpad=0.5,
            columnspacing=1.5
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

        ax.set_title(disp)
        ax.set_xlim(0, 1)
        ax.set_yscale('linear')  # linear explained variance
        ax.grid(True, linestyle=':', linewidth=0.8)

        if col != 0:
            ax.yaxis.set_tick_params(labelleft=False)
        else:
            ax.set_ylabel('Explained variance (%)')

        if row != nrows - 1:
            ax.xaxis.set_tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Components (fraction of max)')

    for j in range(n, len(axs)):
        axs[j].axis('off')

    handles_labels = [ax.get_legend_handles_labels() for ax in axs if hasattr(ax, "get_legend_handles_labels")]
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
            loc='lower center',
            bbox_to_anchor=(0, -0.1, 1, 0.1),
            ncol=min(6, len(labels)),
            mode="expand",
            frameon=True,
            fontsize=28
        )

    cbar = fig.colorbar(
        sm,
        ax=axs.tolist(),
        orientation='vertical',
        fraction=0.04,
        pad=0.04,
        label='Normalized layer number (%)'
    )
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels([f'{int(t*100)}%' for t in np.linspace(0, 1, 5)])

    out_path = os.path.join(out_base, 'multi_variance_by_model.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined variance-by-model plot to {out_path}')

def multi_model_plot(dfs, models, thresholds, max_layers, out_base):
    plot_components_by_threshold_multiplot(dfs, models, thresholds, out_base)
    plot_variance_by_model_multiplot(dfs, models, thresholds, out_base)

def main():
    parser = argparse.ArgumentParser(description='Intrinsic PCA dim per model')
    parser.add_argument('--models', '-M', nargs='+', required=False)
    parser.add_argument('--all', action='store_true', help='Use all models found in output-dir')
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--output-dir', '-o', required=True)
    parser.add_argument(
        '--thresholds', '-t', type=int, nargs='+',
        default=[50, 60, 70, 80, 90, 95, 99]
    )
    parser.add_argument('--max-layers', type=int, default=None)
    parser.add_argument('--override', '-O', action='store_true')
    parser.add_argument('--sanity-check', action='store_true', help='Run sanity check for outliers and massive activations')
    parser.add_argument('--analyze-sanity-check', action='store_true', help='Plot existing sanity check results without re-running analysis')
    parser.add_argument('--collect-3d-stats', action='store_true', help='Collect 3D activation stats (mean abs per dim)')
    parser.add_argument('--plot-3d-stats', action='store_true', help='Plot combined 3D activation surface')
    parser.add_argument('--plot-box-whiskers', action='store_true', help='Plot box-and-whisker plots for layer activations')
    args = parser.parse_args()

    # Determine models list
    if args.all:
        if not os.path.exists(args.output_dir):
            sys.exit(f"Output directory {args.output_dir} does not exist.")
        # Find subdirectories that might be models
        found_models = []
        for d in os.listdir(args.output_dir):
            if os.path.isdir(os.path.join(args.output_dir, d)):
                found_models.append(d)
        if not found_models:
            sys.exit(f"No model directories found in {args.output_dir}")
        args.models = sorted(found_models)
        print(f"Found {len(args.models)} models: {args.models}")
    elif not args.models:
        sys.exit("Error: Must specify --models or --all")

    thresholds = [t / 100.0 for t in args.thresholds]
    
    # Mode: Collect 3D stats
    if args.collect_3d_stats:
        print("Collecting 3D activation stats...")
        for model in args.models:
            out_model = os.path.join(args.output_dir, model)
            collect_activations_3d(model, args.dataset, out_model, override=args.override)
        if args.plot_3d_stats:
            # Fall through to plot if both flags set
            pass
        else:
            return

    # Mode: Plot 3D stats
    if args.plot_3d_stats:
        print("Plotting 3D activation stats...")
        plot_3d_activations(args.models, args.output_dir)
        return

    # Mode: Plot Box Whiskers
    if args.plot_box_whiskers:
        plot_box_whiskers(args.models, args.dataset, args.output_dir, override=args.override, max_layers=args.max_layers)
        return

    dfs, used = [], []
    dfs_dict = {}
    
    if args.analyze_sanity_check:
        print("Analyzing existing sanity check results...")
        for model in args.models:
            out_model = os.path.join(args.output_dir, model)
            report_path = os.path.join(out_model, 'sanity_check_stats.csv')
            if os.path.exists(report_path):
                print(f"Loading {report_path}")
                df = pd.read_csv(report_path)
                plot_sanity_stats(df, model, out_model)
            else:
                print(f"No report found for {model} at {report_path}")
        return

    if args.sanity_check:
        print("Running Sanity Checks (Outliers & Massive Activations)...")
        for model in args.models:
            out_model = os.path.join(args.output_dir, model)
            sanity_check_model(model, args.dataset, out_model)
        return

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
