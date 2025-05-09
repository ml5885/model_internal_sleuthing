import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os
import numpy as np

# Style and DPI
sns.set_style("white")
mpl.rcParams["figure.dpi"] = 150
plt.rcParams.update({"font.size": 12})

palette = sns.color_palette("Set2")

models = [
    "gpt2",
    "qwen2", "qwen2-instruct",
    "pythia1.4b",
    "gemma2b",
    "bert-base-uncased", "bert-large-uncased",
    "deberta-v3-large"
]

model_names = {
    "gpt2": "GPT 2",
    "qwen2": "Qwen 2.5 1.5B",
    "qwen2-instruct": "Qwen 2.5 1.5B-Instruct",
    "pythia1.4b": "Pythia 1.4B",
    "gemma2b": "Gemma 2 2B",
    "bert-base-uncased": "BERT Base Uncased",
    "bert-large-uncased": "BERT Large Uncased",
    "deberta-v3-large": "DeBERTa v3 Large",
}

def get_acc_columns(df, prefix):
    # same as before
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
    n_cols = 3
    n_rows = (len(model_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 6 * n_rows // 2),
                             sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(top=0.93, wspace=0.3, hspace=0.35)

    for idx, model in enumerate(model_list):
        ax = axes[idx]

        # file paths
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

                # line plots
                ax.plot(lex_df["Layer"], lex_sel,
                        label="Lexeme",
                        color=palette[0], linestyle="-", marker="o", markersize=3)
                ax.plot(inf_df["Layer"], inf_sel,
                        label="Inflection",
                        color=palette[1], linestyle="--", marker="x", markersize=4)

                # symmetric y-limits with 10% pad
                allv = np.concatenate([lex_sel.values, inf_sel.values])
                y_ex = max(abs(allv.min()), abs(allv.max()))
                pad = y_ex * 0.1 if y_ex > 0 else 0.1
                ax.set_ylim(-y_ex - pad, y_ex + pad)
                # horizontal pad
                ax.margins(x=0.05)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Missing data",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Selectivity")
        ax.set_title(model_names.get(model, model), fontsize=26)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        if ax.get_legend():
            ax.legend_.remove()

    # Hide unused axes (e.g., bottom right if not enough models)
    for idx in range(len(model_list), len(axes)):
        axes[idx].set_visible(False)

    # consolidated legend in last valid subplot
    legend_ax = axes[len(model_list)-1]
    for ax in axes[:len(model_list)][::-1]:
        h, l = ax.get_legend_handles_labels()
        if h:
            legend_ax.legend(h, l, loc="center", frameon=False, fontsize=12)
            break

    # fig.suptitle("Comparison of Lexeme vs Inflection Selectivity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"selectivity_comparison_{probe_type}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out}")

def plot_probe_advantage(
    task: str,
    model_list: list[str],
    dataset: str,
    output_dir: str = "figures2",
):
    n_cols = 3
    n_rows = (len(model_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 6 * n_rows // 2),
                             sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(top=0.93, wspace=0.3, hspace=0.35)

    for idx, model in enumerate(model_list):
        ax = axes[idx]

        lin_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_reg", f"{task}_results.csv")
        mlp_csv = os.path.join(f"../output/probes/{dataset}_{model}_{task}_mlp", f"{task}_results.csv")

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

                ax.bar(common, adv,
                       color=palette[2], alpha=0.7, width=0.7)
                ax.axhline(0, linestyle="--", color="gray")

                # symmetric y-limits with 10% pad
                y_ex = max(abs(adv.min()), abs(adv.max()))
                pad = y_ex * 0.1 if y_ex > 0 else 0.1
                ax.set_ylim(-y_ex - pad, y_ex + pad)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Missing data",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Layer")
        ax.set_ylabel("MLP Advantage (pp)")
        ax.set_title(model_names.get(model, model), fontsize=26)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes (e.g., bottom right if not enough models)
    for idx in range(len(model_list), len(axes)):
        axes[idx].set_visible(False)

    # fig.suptitle(f"MLP vs Linear Probe Advantage for {task.capitalize()} Prediction", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"mlp_advantage_{task}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out}")

def create_peak_layer_table(
    model_list, dataset, probe_type="reg", output_dir="figures2"
):
    # identical to yours
    results = []
    for model in model_list:
        mr = {"Model": model_names.get(model, model)}
        # lexeme
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
        # inflection
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
        # gap
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

    # format
    for col in ["Lexeme Peak Acc", "Inflection Peak Acc"]:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        mask = df[col] != "N/A"
        df.loc[mask, col] = df.loc[mask, col].map(lambda x: f"{float(x):.3f}")

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

# Run all
dataset = "ud_gum_dataset"
os.makedirs("figures2", exist_ok=True)

plot_selectivity_comparison(models, dataset, probe_type="reg")
plot_selectivity_comparison(models, dataset, probe_type="mlp")
plot_probe_advantage("lexeme", models, dataset)
plot_probe_advantage("inflection", models, dataset)
create_peak_layer_table(models, dataset, probe_type="mlp")