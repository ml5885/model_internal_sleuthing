import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np

sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 24,
    "axes.titlesize": 30,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "legend.title_fontsize": 24,
    "axes.linewidth": 1.5,
    "grid.linewidth": 1.0
})

PLOT_CONFIG = {
    'figure_width': 4.8,
    'figure_height': 3.5,
    'n_cols': 4,
    'wspace': 0.1,
    'hspace': 0.25,
    'top_margin': 0.93,
    'legend_y_offset': -0.02
}

bbox_to_anchor = (0, -0.13, 1, 0.1)
palette = sns.color_palette("Set2")

models = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl", "qwen2", "qwen2-instruct", "gemma2b",
    "gemma2b-it", "llama3-8b", "llama3-8b-instruct", "pythia-6.9b",
    "pythia-6.9b-tulu", "olmo2-7b-instruct", "olmo2-7b"
]

model_names = {
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "pythia1.4b": "Pythia-1.4B",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "bert-base-uncased": "BERT-Base-Uncased",
    "bert-large-uncased": "BERT-Large-Uncased",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
}

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

def plot_selectivity_comparison(
    model_list: list[str],
    dataset: str,
    probe_type: str = "reg",
    output_dir: str = "figures2",
):
    n_cols = PLOT_CONFIG['n_cols']
    n_rows = (len(model_list) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PLOT_CONFIG['figure_width'] * n_cols, 
                                    PLOT_CONFIG['figure_height'] * n_rows),
                             sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(top=PLOT_CONFIG['top_margin'], 
                       wspace=PLOT_CONFIG['wspace'], 
                       hspace=PLOT_CONFIG['hspace'])
    handles, labels = None, None
    
    global_min = float('inf')
    global_max = float('-inf')
    
    data_for_plots = []
    
    for idx, model in enumerate(model_list):
        ax = axes[idx]

        lex_csv = os.path.join(f"../output/probes/{dataset}_{model}_lexeme_{probe_type}", "lexeme_results.csv")
        inf_csv = os.path.join(f"../output/probes/{dataset}_{model}_inflection_{probe_type}", "inflection_results.csv")

        if os.path.exists(lex_csv) and os.path.exists(inf_csv):
            lex_df = pd.read_csv(lex_csv)
            inf_df = pd.read_csv(inf_csv)
            try:
                lac, lcc = get_acc_columns(lex_df, "lexeme")
                iac, icc = get_acc_columns(inf_df, "inflection")
                lex_sel = lex_df[lac] - lex_df[lcc]
                inf_sel = inf_df[iac] - inf_df[icc]

                def norm_layers(df):
                    layers = df["Layer"].values
                    min_layer = layers.min()
                    max_layer = layers.max()
                    if max_layer == min_layer:
                        return np.zeros_like(layers, dtype=float)
                    return (layers - min_layer) / (max_layer - min_layer)

                lex_norm = norm_layers(lex_df)
                inf_norm = norm_layers(inf_df)
                
                global_min = min(global_min, lex_sel.min(), inf_sel.min())
                global_max = max(global_max, lex_sel.max(), inf_sel.max())
                
                data_for_plots.append({
                    'lex_norm': lex_norm,
                    'inf_norm': inf_norm,
                    'lex_sel': lex_sel,
                    'inf_sel': inf_sel,
                    'valid': True
                })
            except Exception as e:
                data_for_plots.append({
                    'error': str(e),
                    'valid': False
                })
        else:
            data_for_plots.append({
                'error': "Missing data",
                'valid': False
            })
    
    if global_min != float('inf') and global_max != float('-inf'):
        y_range = global_max - global_min
        y_padding = y_range * 0.1
        y_min = global_min - y_padding
        y_max = global_max + y_padding
    else:
        y_min, y_max = -0.5, 1.0

    for idx, (model, plot_data) in enumerate(zip(model_list, data_for_plots)):
        ax = axes[idx]

        if plot_data['valid']:
            ax.plot(plot_data['lex_norm'], plot_data['lex_sel'],
                    label="Lexeme",
                    color=palette[0], linestyle="-", marker="o", markersize=3)
            ax.plot(plot_data['inf_norm'], plot_data['inf_sel'],
                    label="Inflection",
                    color=palette[1], linestyle="--", marker="x", markersize=4)

            ax.set_ylim(y_min, y_max)
            ax.margins(x=0.05)

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.set_xlim(0, 1)
            xticks = np.array([0, 0.5, 1.0])
            row, col = divmod(idx, n_cols)
            if row == n_rows - 1:
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])
            else:
                ax.set_xticks([])
        else:
            ax.text(0.5, 0.5, plot_data['error'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])

        row, col = divmod(idx, n_cols)
        if col == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
        if row == n_rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")
        ax.set_title(model_names.get(model, model))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        if ax.get_legend():
            ax.legend_.remove()

    for idx in range(len(model_list), len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles, labels, loc="lower center",
               ncol=2,
               bbox_to_anchor=(0.5, PLOT_CONFIG['legend_y_offset']),
               frameon=True)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"selectivity_comparison_{probe_type}.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure to {out}")

def plot_probe_advantage(
    task: str,
    model_list: list[str],
    dataset: str,
    output_dir: str = "figures2",
):
    n_cols = PLOT_CONFIG['n_cols']
    n_rows = (len(model_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PLOT_CONFIG['figure_width'] * n_cols, 
                                    PLOT_CONFIG['figure_height'] * n_rows),
                             sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(top=PLOT_CONFIG['top_margin'], 
                       wspace=PLOT_CONFIG['wspace'], 
                       hspace=PLOT_CONFIG['hspace'])

    global_min = float('inf')
    global_max = float('-inf')
    
    plot_data_list = []
    
    for idx, model in enumerate(model_list):
        ax = axes[idx]

        lin_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_reg", f"{task}_results.csv")
        mlp_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_mlp", f"{task}_results.csv")
        
        if not os.path.exists(lin_csv):
            lin_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_linear", f"{task}_results.csv")
        if not os.path.exists(mlp_csv):
            mlp_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_nonlinear", f"{task}_results.csv")
            
        if not os.path.exists(mlp_csv):
            mlp_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_nn", f"{task}_results.csv")

        if os.path.exists(lin_csv) and os.path.exists(mlp_csv):
            lin_df = pd.read_csv(lin_csv)
            mlp_df = pd.read_csv(mlp_csv)
            try:
                lac, _ = get_acc_columns(lin_df, task)
                mac, _ = get_acc_columns(mlp_df, task)

                common = np.intersect1d(lin_df["Layer"], mlp_df["Layer"])
                lf = lin_df[lin_df["Layer"].isin(common)].sort_values("Layer")
                mf = mlp_df[mlp_df["Layer"].isin(common)].sort_values("Layer")
                adv = mf[mac].values - lf[lac].values

                layers = lf["Layer"].values
                min_layer = layers.min()
                max_layer = layers.max()
                
                global_min = min(global_min, adv.min())
                global_max = max(global_max, adv.max())
                
                plot_data_list.append({
                    'layers': layers,
                    'adv': adv,
                    'min_layer': min_layer,
                    'max_layer': max_layer,
                    'valid': True
                })
            except Exception as e:
                plot_data_list.append({
                    'error': f"Error: {e}",
                    'valid': False
                })
        else:
            missing_files = []
            if not os.path.exists(lin_csv):
                missing_files.append(f"Linear: {os.path.basename(lin_csv)}")
            if not os.path.exists(mlp_csv):
                missing_files.append(f"MLP: {os.path.basename(mlp_csv)}")
            
            plot_data_list.append({
                'error': f"Missing files:\n{chr(10).join(missing_files)}",
                'valid': False
            })

    if global_min != float('inf') and global_max != float('-inf'):
        y_range = global_max - global_min
        if y_range == 0:
            y_padding = 0.1
        else:
            y_padding = y_range * 0.15  # Increased padding
        
        y_min = global_min - y_padding
        y_max = global_max + y_padding
        
        # Ensure we include zero in the range for reference
        if global_min > 0:
            y_min = min(y_min, -y_padding)
        if global_max < 0:
            y_max = max(y_max, y_padding)
    else:
        y_min, y_max = -0.2, 0.2

    for idx, (model, plot_data) in enumerate(zip(model_list, plot_data_list)):
        ax = axes[idx]

        if plot_data['valid']:
            layers = plot_data['layers']
            adv = plot_data['adv']
            
            ax.bar(layers, adv, color=palette[2], alpha=0.7, width=0.8)
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(plot_data['min_layer'] - 0.5, plot_data['max_layer'] + 0.5)
            
            row, col = divmod(idx, n_cols)
            if row == n_rows - 1:
                ax.set_xticks(layers[::max(1, len(layers)//5)])  # Show max 5 ticks
            else:
                ax.set_xticks([])
        else:
            ax.text(0.5, 0.5, plot_data['error'],
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14)
            ax.set_xticks([])

        row, col = divmod(idx, n_cols)
        if col == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
        if row == n_rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")
        ax.set_title(model_names.get(model, model))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    for idx in range(len(model_list), len(axes)):
        axes[idx].set_visible(False)
        
    fig.text(-0.015, 0.5, 'MLP Advantage', va='center', ha='center', 
                rotation=90, fontsize=28)
    
    fig.text(0.5, -0.015, 'Normalized layer number (%)', va='center', ha='center', 
                fontsize=28)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"mlp_advantage_{task}.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure to {out}")

def create_peak_layer_table(
    model_list, dataset, probe_type="reg", output_dir="figures2"
):
    results = []
    for model in model_list:
        mr = {"Model": model_names.get(model, model)}
        lex_csv = os.path.join(f"../output/probes/{dataset}_{model}_lexeme_{probe_type}", "lexeme_results.csv")
        if os.path.exists(lex_csv):
            df = pd.read_csv(lex_csv)
            try:
                ac, _ = get_acc_columns(df, "lexeme")
                idx = df[ac].idxmax()
                mr["Lexeme Peak Layer"] = df.loc[idx, "Layer"]
                mr["Lexeme Peak Acc"] = df.loc[idx, ac]
            except:
                mr["Lexeme Peak Layer"] = "N/A"
                mr["Lexeme Peak Acc"] = "N/A"
        else:
            mr["Lexeme Peak Layer"] = "N/A"
            mr["Lexeme Peak Acc"] = "N/A"
        inf_csv = os.path.join(f"../output/probes/{dataset}_{model}_inflection_{probe_type}", "inflection_results.csv")
        if os.path.exists(inf_csv):
            df = pd.read_csv(inf_csv)
            try:
                ac, _ = get_acc_columns(df, "inflection")
                idx = df[ac].idxmax()
                mr["Inflection Peak Layer"] = df.loc[idx, "Layer"]
                mr["Inflection Peak Acc"] = df.loc[idx, ac]
            except:
                mr["Inflection Peak Layer"] = "N/A"
                mr["Inflection Peak Acc"] = "N/A"
        else:
            mr["Inflection Peak Layer"] = "N/A"
            mr["Inflection Peak Acc"] = "N/A"
        if (
            mr["Lexeme Peak Layer"] != "N/A"
            and mr["Inflection Peak Layer"] != "N/A"
        ):
            mr["Layer Gap"] = float(mr["Lexeme Peak Layer"]) - float(mr["Inflection Peak Layer"])
        else:
            mr["Layer Gap"] = "N/A"
        results.append(mr)

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"peak_layer_summary_{probe_type}.csv")
    df.to_csv(csv_path, index=False)

    for col in ["Lexeme Peak Acc", "Inflection Peak Acc"]:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = ~df[col].isna()
            df[col] = df[col].astype(object)
            df.loc[mask, col] = df.loc[mask, col].map(lambda x: f"{float(x):.3f}")
        except ValueError:
            pass

    tex_path = os.path.join(output_dir, f"peak_layer_summary_{probe_type}.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Peak performance layers for lemma and inflection prediction across models.}\n")
        f.write("\\label{tab:peak_layers}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Lexeme Peak & Lexeme Peak & Inflection Peak & Inflection Peak \\\\\n")
        f.write(" & Layer & Accuracy & Layer & Accuracy \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Model']} & {row['Lexeme Peak Layer']} & {row['Lexeme Peak Acc']} & ")
            f.write(f"{row['Inflection Peak Layer']} & {row['Inflection Peak Acc']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved table to {csv_path} and {tex_path}")
    return df

dataset = "ud_gum_dataset"
os.makedirs("figures2", exist_ok=True)

plot_selectivity_comparison(models, dataset, probe_type="reg")
plot_selectivity_comparison(models, dataset, probe_type="nn")
plot_probe_advantage("lexeme", models, dataset)
plot_probe_advantage("inflection", models, dataset)
create_peak_layer_table(models, dataset, probe_type="nn")
