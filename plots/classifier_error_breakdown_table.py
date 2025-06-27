import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

MODELS = [
    "bert-base-uncased", "bert-large-uncased", "deberta-v3-large",
    "gpt2", "gpt2-large", "gpt2-xl",
    "pythia-6.9b", "pythia-6.9b-tulu",
    "olmo2-7b", "olmo2-7b-instruct",
    "gemma2b", "gemma2b-it",
    "qwen2", "qwen2-instruct",
    "llama3-8b", "llama3-8b-instruct"
]
MODEL_DISPLAY_NAMES = {
    "bert-base-uncased": "BERT-Base",
    "bert-large-uncased": "BERT-Large",
    "deberta-v3-large": "DeBERTa-v3-Large",
    "gpt2": "GPT-2-Small",
    "gpt2-large": "GPT-2-Large",
    "gpt2-xl": "GPT-2-XL",
    "qwen2": "Qwen2.5-1.5B",
    "qwen2-instruct": "Qwen2.5-1.5B-Instruct",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B"
}
PROBE_TYPES = ["reg", "nn"]
DATASET = "ud_gum_dataset"
OUTPUT_DIR = "../output/probes"
LANG_SUBDIR = DATASET if DATASET not in ("ud_gum_dataset", "en_gum", "english", "en_gum_dataset") else None

CONLLU_PATH = "../data/en_gum-ud-train.conllu"
POS_CACHE_PATH = "../output/lemma_to_pos_cache.json"

ORDERED_POS_GROUPS = ["Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Preposition", "Conjunction", "Interjection", "Other"]
UPOS_TO_TARGET_GROUP = {
    "NOUN": "Noun", "PROPN": "Noun", 
    "VERB": "Verb", "AUX": "Verb",   
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PRON": "Pronoun",
    "ADP": "Preposition", 
    "CCONJ": "Conjunction", "SCONJ": "Conjunction", 
    "INTJ": "Interjection",
    
    "DET": "Other", "NUM": "Other", "PART": "Other",
    "PUNCT": "Other", "SYM": "Other", "X": "Other"
}

CANONICAL_INFLECTION_KEYS = [
    "3rd_pers", "base", "comparative", "past", "plural", "positive", "singular", "superlative"
]

def get_pred_path(model, task, probe):
    if LANG_SUBDIR:
        return os.path.join(
            "../output", LANG_SUBDIR, "probes",
            f"{DATASET}_{model}_{task}_{probe}",
            "predictions.csv"
        )
    else:
        return os.path.join(
            OUTPUT_DIR,
            f"{DATASET}_{model}_{task}_{probe}",
            "predictions.csv"
        )

def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def load_or_create_lemma_pos_mapping(conllu_path, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    lemma_to_pos = {}
    print(f"Parsing {conllu_path} to create lemma-POS mapping...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4: 
                lemma = parts[2].lower() 
                upos = parts[3]          
                if lemma and upos and lemma != "_": 
                    
                    lemma_to_pos[lemma] = upos
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_to_pos, f)
    print(f"Lemma-POS mapping cached to {cache_path}")
    return lemma_to_pos

def get_grouped_accuracy(df):
    if "y_true_str" not in df.columns or "y_pred_str" not in df.columns:
        return None, None

    counts = df.groupby("y_true_str").size()
    accs = df.groupby("y_true_str")["y_pred_str"].apply(
        lambda preds: np.mean(preds.values == df.loc[preds.index, "y_true_str"].values)
    )
    return accs, counts

results = []
missing = []

INFLECTION_DISPLAY = {
    "3rd_pers": "3rd person", "base": "Base", "comparative": "Comparative",
    "past": "Past", "plural": "Plural", "positive": "Positive",
    "singular": "Singular", "superlative": "Superlative"
}
PROBE_DISPLAY = {"reg": "Linear", "nn": "MLP"}

for model in tqdm(MODELS, desc="Models"):
    for probe in PROBE_TYPES:
        path = get_pred_path(model, "inflection", probe)
        df = safe_read_csv(path)
        if df is None:
            missing.append((model, probe, path))
            continue
        accs, counts = get_grouped_accuracy(df)
        if accs is None:
            missing.append((model, probe, path))
            continue
        for grp, acc in accs.items():
            results.append({
                "Model": model,
                "Probe": probe,
                "Group": grp,
                "N": counts[grp],
                "Accuracy": acc
            })

ORDERED_INFLECTION_GROUPS_INTERNAL = [k for k in CANONICAL_INFLECTION_KEYS if k in INFLECTION_DISPLAY]

existing = {(r['Model'], r['Probe'], r['Group']) for r in results}
for model in MODELS:
    for probe in PROBE_TYPES:
        for grp in ORDERED_INFLECTION_GROUPS_INTERNAL:
            if (model, probe, grp) not in existing:
                results.append({
                    "Model": model,
                    "Probe": probe,
                    "Group": grp,
                    "N": 0,
                    "Accuracy": np.nan
                })

res_df = pd.DataFrame(results)

inflection_cols = pd.MultiIndex.from_product(
    [[INFLECTION_DISPLAY[g] for g in ORDERED_INFLECTION_GROUPS_INTERNAL], [PROBE_DISPLAY[p] for p in PROBE_TYPES]],
    names=["Inflection", "Probe"]
)
table_rows = []
for model in MODELS:
    row = []
    for grp in ORDERED_INFLECTION_GROUPS_INTERNAL:
        for pr in PROBE_TYPES:
            m = res_df[(res_df.Model == model) & (res_df.Probe == pr) & (res_df.Group == grp)]
            val = m.Accuracy.iloc[0] if not m.empty else np.nan
            row.append(f"{val*100:.1f}" if not np.isnan(val) else "--")
    table_rows.append(row)
model_display_names = [MODEL_DISPLAY_NAMES[m] for m in MODELS]
inflection_df = pd.DataFrame(table_rows, index=model_display_names, columns=inflection_cols)

canonical_inflection_df_for_counts = None
preferred_probe_order_for_counts = ["reg", "nn"]

if MODELS and PROBE_TYPES:
    for model_name_for_counts in MODELS:
        for probe_type_for_counts in preferred_probe_order_for_counts:
            if probe_type_for_counts not in PROBE_TYPES:
                continue

            path_for_counts = get_pred_path(model_name_for_counts, "inflection", probe_type_for_counts)
            df_candidate_for_counts = safe_read_csv(path_for_counts)
            
            if df_candidate_for_counts is not None and \
               "y_true_str" in df_candidate_for_counts.columns and \
               not df_candidate_for_counts.empty:
                
                canonical_inflection_df_for_counts = df_candidate_for_counts
                print(f"INFO: Using '{path_for_counts}' for inflection group counts.")
                break
        if canonical_inflection_df_for_counts is not None:
            break

inflection_group_counts = {}
if canonical_inflection_df_for_counts is not None:
    if 'layer' in canonical_inflection_df_for_counts.columns and not canonical_inflection_df_for_counts.empty:
        representative_layer = canonical_inflection_df_for_counts['layer'].min()
        single_layer_df = canonical_inflection_df_for_counts[canonical_inflection_df_for_counts['layer'] == representative_layer]
        print(f"INFO: For inflection counts, using layer {representative_layer} from canonical file. Shape: {single_layer_df.shape}")
        counts_from_canonical_inflection = single_layer_df.groupby("y_true_str").size()
    else:
        print("WARNING: 'layer' column not found in canonical inflection file or file empty after check. Using all data for counts.")
        counts_from_canonical_inflection = canonical_inflection_df_for_counts.groupby("y_true_str").size()
    
    for grp_key in ORDERED_INFLECTION_GROUPS_INTERNAL:
        display_name = INFLECTION_DISPLAY.get(grp_key)
        if display_name:
            count = counts_from_canonical_inflection.get(grp_key, 0)
            inflection_group_counts[display_name] = int(count)
else:
    print("WARNING: No canonical inflection predictions file found. Inflection group counts will be zero.")
    for grp_key in ORDERED_INFLECTION_GROUPS_INTERNAL:
        display_name = INFLECTION_DISPLAY.get(grp_key, grp_key)
        inflection_group_counts[display_name] = 0

lemma_pos_map = load_or_create_lemma_pos_mapping(CONLLU_PATH, POS_CACHE_PATH)

canonical = None
for model in MODELS:
    for pr in PROBE_TYPES:
        df = safe_read_csv(get_pred_path(model, "lexeme", pr))
        if df is not None and "y_true_str" in df.columns:
            canonical = df.copy()
            break
    if canonical is not None:
        break

pos_group_counts = {group: 0 for group in ORDERED_POS_GROUPS}

if canonical is not None:
    # Use data from a single representative layer for unique lexeme counts for POS grouping
    if 'layer' in canonical.columns and not canonical.empty:
        representative_layer_lex = canonical['layer'].min()
        canonical_single_layer = canonical[canonical['layer'] == representative_layer_lex]
        print(f"INFO: For POS group counts, using layer {representative_layer_lex} from canonical lexeme file. Shape: {canonical_single_layer.shape}")
        unique_lexemes_in_dataset = canonical_single_layer.y_true_str.str.lower().unique()
    else:
        print("WARNING: 'layer' column not found in canonical lexeme file or file empty. Using all data for unique lexemes.")
        unique_lexemes_in_dataset = canonical.y_true_str.str.lower().unique()

    for lexeme in unique_lexemes_in_dataset:
        upos = lemma_pos_map.get(lexeme) 
        pos_group = UPOS_TO_TARGET_GROUP.get(upos, "Other") if upos else "Other"
        
        if pos_group in pos_group_counts:
            pos_group_counts[pos_group] += 1
        else:
            pos_group_counts["Other"] += 1 


    pos_results = [] 
    for model in tqdm(MODELS, desc="Models (lexeme POS groups)"): 
        for pr in PROBE_TYPES:
            df = safe_read_csv(get_pred_path(model, "lexeme", pr))
            if df is None or "y_true_str" not in df.columns or "y_pred_str" not in df.columns:
                for pos_g in ORDERED_POS_GROUPS: 
                    pos_results.append({
                        "Model": model,
                        "Probe": pr,
                        "POS_Group": pos_g, 
                        "Accuracy": np.nan
                    })
                continue
            
            df['pos_group'] = df.y_true_str.str.lower().apply(
                lambda lem: UPOS_TO_TARGET_GROUP.get(lemma_pos_map.get(lem), "Other")
            )

            for pos_g in ORDERED_POS_GROUPS: 
                sub = df[df.pos_group == pos_g]
                acc = np.mean(sub.y_pred_str == sub.y_true_str) if len(sub) > 0 else np.nan
                pos_results.append({
                    "Model": model,
                    "Probe": pr,
                    "POS_Group": pos_g, 
                    "Accuracy": acc
                })

    lex_df_data = pd.DataFrame(pos_results) 

    lexeme_cols_for_table = pd.MultiIndex.from_product(
        [ORDERED_POS_GROUPS, [PROBE_DISPLAY[p] for p in PROBE_TYPES]],
        names=["Part of Speech", "Probe"] 
    )
    lex_rows_for_table = []
    for model_id in MODELS: 
        row = []
        for pos_g in ORDERED_POS_GROUPS: 
            for pr_type in PROBE_TYPES:
                m = lex_df_data[(lex_df_data.Model == model_id) & (lex_df_data.Probe == pr_type) & (lex_df_data.POS_Group == pos_g)]
                val = m.Accuracy.values[0] if not m.empty and not pd.isna(m.Accuracy.values[0]) else np.nan
                # Format as percent with one decimal
                row.append(f"{val*100:.1f}" if not np.isnan(val) else "--")
        lex_rows_for_table.append(row)
    
    lexeme_df = pd.DataFrame(lex_rows_for_table, index=model_display_names, columns=lexeme_cols_for_table)
else:
    empty_cols = pd.MultiIndex.from_product(
        [ORDERED_POS_GROUPS, [PROBE_DISPLAY[p] for p in PROBE_TYPES]],
        names=["Part of Speech", "Probe"]
    )
    lexeme_df = pd.DataFrame(columns=empty_cols)

def to_latex_table(df, caption_text, label, group_counts_map): 
    if df.columns.empty:
        return f"% No data columns available for table: {caption_text}\n% DataFrame was empty or had no columns.\n"

    group_names_from_df = df.columns.unique(level=0) 
    num_groups = len(group_names_from_df)
    num_probes_per_group = len(PROBE_TYPES) if num_groups > 0 else 0


    n_row_formatted = []
    for group_name in group_names_from_df: 
        count_val = group_counts_map.get(str(group_name)) 
        
        if count_val is not None:
            n_row_formatted.append(f"{count_val:,}")
        else:
            n_row_formatted.append("--")

    latex = (
        '\\begin{table*}[t]\n'
        '\\small\n'
        '\\centering\n'
        '\\renewcommand\\arraystretch{1.2}\n'
        '\\setlength{\\tabcolsep}{10pt}\n'
        '\\begin{tabular}{@{}l' + 'c' * (num_groups * num_probes_per_group) + '@{}}\n'
        '    \\toprule\n'
        '    \\multirow{3}{*}{\\textbf{Model}}'
    )

    for group_name_display in group_names_from_df:
        latex += f" & \\multicolumn{{{num_probes_per_group}}}{{c}}{{{group_name_display}}}"
    latex += " \\\\\n"

    for n_str in n_row_formatted:
        latex += f" & \\multicolumn{{{num_probes_per_group}}}{{c}}{{\\footnotesize (n={n_str})}}"
    latex += " \\\\\n"

    cmidrules = []
    for i in range(num_groups):
        start = 2 + i * num_probes_per_group
        end = start + num_probes_per_group - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    latex += "    " + " ".join(cmidrules) + "\n"

    probe_labels = [PROBE_DISPLAY[p] for p in PROBE_TYPES]
    for _ in group_names_from_df:
        for probe_lbl in probe_labels:
            latex += f" & {probe_lbl}"
    latex += " \\\\\n"
    latex += "    \\midrule\n"

    for idx, row_values in zip(df.index, df.values):
        latex += f"    {idx}"
        for val in row_values:
            latex += f" & {val}"
        latex += " \\\\\n"
    latex += "    \\bottomrule\n"
    latex += "  \\end{tabular}\n"

    footnote = (
        "All values represent classification accuracy on a 0--1 scale."
    )
    latex += f"\\caption{{{caption_text} {footnote}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\end{table*}\n"
    return latex

def to_latex_single_probe_table(df, probe_key, caption_text, label, group_counts_map=None):
    if df.columns.empty:
        return f"% No data columns available for table: {caption_text}\n% DataFrame had no columns.\n"

    try:
        probe_df = df.xs(PROBE_DISPLAY[probe_key], axis=1, level="Probe")
    except KeyError:
        if "Probe" in df.columns.names: 
             return f"% Probe type '{PROBE_DISPLAY[probe_key]}' not found in columns for table: {caption_text}\n"
        probe_df = df 
        if group_counts_map is not None: 
            pass

    latex = (
        '\\begin{table*}[t]\n'
        '\\small\n'
        '\\centering\n'
        '\\renewcommand\\arraystretch{1.2}\n'
        '\\setlength{\\tabcolsep}{3pt}\n' 
        '\\begin{minipage}{\\linewidth}\\centering\n'
        '\\begin{tabular}{@{}l' + 'c' * probe_df.shape[1] + '@{}}\n'
        '    \\toprule\n'
    )
    
    header_row1 = '    \\multirow{2}{*}{\\textbf{Model}}' if group_counts_map else '    \\textbf{Model}'
    for col_name in probe_df.columns:
        header_row1 += f" & {col_name}"
    header_row1 += " \\\\\n"
    latex += header_row1

    if group_counts_map:
        header_row2 = '    ' 
        for col_name in probe_df.columns:
            count = group_counts_map.get(str(col_name), "--") 
            n_str = f"{count:,}" if isinstance(count, int) else str(count)
            header_row2 += f" & \\footnotesize (n={n_str})"
        header_row2 += " \\\\\n"
        latex += header_row2
        latex += "    \\midrule\n" 
    else:
        latex += "    \\midrule\n"


    for idx, row_values in zip(probe_df.index, probe_df.values):
        latex += f"    {idx}"
        for val in row_values:
            latex += f" & {val}"
        latex += " \\\\\n"
    latex += "    \\bottomrule\n"
    latex += "\\end{tabular}\n"
    
    caption_suffix = (
        " Accuracies are calculated over all examples for a given group across all layers. "
        "Counts (n) are derived from a single representative layer for each group. "
        "All accuracy values are percentages."
    )
    latex += f"\\caption{{{caption_text}{caption_suffix}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\end{minipage}\n"
    latex += "\\end{table*}\n"
    return latex

inflection_caption_lr = (
    "Breakdown of inflection classification accuracy for each model by inflection type using Linear Regression classifiers."
)
inflection_caption_mlp = (
    "Breakdown of inflection classification accuracy for each model by inflection type using Multi-Layer Perceptron (MLP) classifiers."
)
lexeme_caption_pos_lr = ( 
    "Breakdown of lexeme classification accuracy by Part of Speech (POS) for each model, using Linear Regression classifiers. "
    "Lexemes are grouped by their POS tags (e.g., Noun, Verb, Adjective)."
)
lexeme_caption_pos_mlp = ( 
    "Breakdown of lexeme classification accuracy by Part of Speech (POS) for each model, using Multi-Layer Perceptron (MLP) classifiers. "
    "Lexemes are grouped by their POS tags (e.g., Noun, Verb, Adjective)."
)


print(to_latex_single_probe_table(inflection_df, "reg", inflection_caption_lr, "tab:inflection_breakdown_lr", inflection_group_counts))
print(to_latex_single_probe_table(inflection_df, "nn", inflection_caption_mlp, "tab:inflection_breakdown_mlp", inflection_group_counts))

print(to_latex_single_probe_table(lexeme_df, "reg", lexeme_caption_pos_lr, "tab:lexeme_pos_breakdown_lr", pos_group_counts))
print(to_latex_single_probe_table(lexeme_df, "nn", lexeme_caption_pos_mlp, "tab:lexeme_pos_breakdown_mlp", pos_group_counts))
