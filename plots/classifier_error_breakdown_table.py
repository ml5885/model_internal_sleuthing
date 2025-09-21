import os
import pandas as pd
import numpy as np
# from tqdm import tqdm
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
    "qwen2.5-7B": "Qwen2.5-7B",
    "qwen2.5-7B-instruct": "Qwen2.5-7B-Instruct",
    "gemma2b": "Gemma-2-2B",
    "gemma2b-it": "Gemma-2-2B-Instruct",
    "llama3-8b": "Llama-3-8B",
    "llama3-8b-instruct": "Llama-3-8B-Instruct",
    "pythia-6.9b": "Pythia-6.9B",
    "pythia-6.9b-tulu": "Pythia-6.9B-Tulu",
    "olmo2-7b-instruct": "OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "OLMo-2-1124-7B",
    "mt5": "mT5-Base",
    "byt5": "ByT5-Base",
    "goldfish_eng_latn_1000mb": "Goldfish English",
    "goldfish_zho_hans_1000mb": "Goldfish Chinese",
    "goldfish_deu_latn_1000mb": "Goldfish German",
    "goldfish_fra_latn_1000mb": "Goldfish French",
    "goldfish_rus_cyrl_1000mb": "Goldfish Russian",
    "goldfish_tur_latn_1000mb": "Goldfish Turkish",
}
PROBE_TYPES = ["reg", "nn"]
OUTPUT_DIR = "../output/probes"

def get_conllu_path_from_dataset(dataset_name):
    """
    Map dataset name to corresponding CoNLL-U file path.
    """
    mapping = {
        "ud_gum_dataset": "../data/en_gum-ud-train.conllu",
        "ud_zh_gsd_dataset": "../data/zh_gsd-ud-train.conllu", 
        "ud_de_gsd_dataset": "../data/de_gsd-ud-train.conllu",
        "ud_fr_gsd_dataset": "../data/fr_gsd-ud-train.conllu",
        "ud_ru_syntagrus_dataset": "../data/ru_syntagrus-ud-train-a.conllu",
        "ud_tr_imst_dataset": "../data/tr_imst-ud-train.conllu",
    }
    return mapping.get(dataset_name)

def get_pos_cache_path(dataset_name):
    """Get cache path for POS mappings for a specific dataset."""
    return f"../output/lemma_to_pos_cache_{dataset_name}.json"

def get_lemma_pos_cache_path(dataset_name):
    """Get cache path for lemma-to-POS mappings for a specific dataset."""
    return f"../output/lemma_pos_mapping_cache_{dataset_name}.json"

def get_language_name_from_dataset(dataset_name):
    """Map dataset name to human-readable language name."""
    mapping = {
        "ud_gum_dataset": "English",
        "ud_zh_gsd_dataset": "Chinese", 
        "ud_de_gsd_dataset": "German",
        "ud_fr_gsd_dataset": "French",
        "ud_ru_syntagrus_dataset": "Russian",
        "ud_tr_imst_dataset": "Turkish",
    }
    return mapping.get(dataset_name, dataset_name)

def parse_conllu_for_pos(conllu_path):
    """
    Parse CoNLL-U file to extract UPOS tags and create frequency counts.
    Returns dictionary mapping UPOS to frequency.
    """
    if not os.path.exists(conllu_path):
        # print(f"Warning: CoNLL-U file not found: {conllu_path}")
        return {}
    
    upos_counts = {}
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse token lines (should have 10 tab-separated fields)
            fields = line.split('\t')
            if len(fields) >= 4:
                # Field 3 is UPOS (universal POS tag)
                upos = fields[3]
                if upos and upos != '_':
                    upos_counts[upos] = upos_counts.get(upos, 0) + 1
    
    return upos_counts

def create_pos_groups_from_data(upos_counts):
    """
    Create POS groups based on the actual UPOS tags found in the data.
    Returns both the ordered group list and the mapping dictionary.
    """
    # Universal POS tag groupings that should work across languages
    base_mapping = {
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
    
    # Only include groups that actually appear in the data
    found_groups = set()
    upos_to_group = {}
    
    for upos in upos_counts.keys():
        if upos in base_mapping:
            group = base_mapping[upos]
            found_groups.add(group)
            upos_to_group[upos] = group
        else:
            # Unknown UPOS tags go to "Other"
            found_groups.add("Other")
            upos_to_group[upos] = "Other"
    
    # Order groups by importance/frequency
    group_order = ["Noun", "Verb", "Adjective", "Adverb", "Pronoun", 
                   "Preposition", "Conjunction", "Interjection", "Other"]
    
    # Only include groups that were actually found
    ordered_groups = [g for g in group_order if g in found_groups]
    
    return ordered_groups, upos_to_group

def get_pos_info_for_dataset(dataset_name):
    """
    Get POS information for a specific dataset.
    Returns (ordered_pos_groups, upos_to_target_group).
    """
    conllu_path = get_conllu_path_from_dataset(dataset_name)
    cache_path = get_pos_cache_path(dataset_name)
    
    if not conllu_path:
        # print(f"Warning: No CoNLL-U mapping found for dataset {dataset_name}")
        # Return default English mapping as fallback
        return ["Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Preposition", "Conjunction", "Interjection", "Other"], {
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
    
    # Check if we have cached POS info
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data['ordered_groups'], cached_data['upos_mapping']
        except:
            # print(f"Warning: Could not load cached POS data from {cache_path}")
            pass
    
    # Parse CoNLL-U file to get POS information
    # print(f"Analyzing POS tags in {conllu_path}...")
    upos_counts = parse_conllu_for_pos(conllu_path)
    
    if not upos_counts:
        # print(f"Warning: No UPOS tags found in {conllu_path}")
        # Return default mapping
        return get_pos_info_for_dataset(None)  # Will trigger fallback
    
    ordered_groups, upos_mapping = create_pos_groups_from_data(upos_counts)
    
    # Cache the results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'ordered_groups': ordered_groups,
        'upos_mapping': upos_mapping,
        'upos_counts': upos_counts
    }
    
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        # print(f"Cached POS information to {cache_path}")
    except Exception as e:
        # print(f"Warning: Could not cache POS data: {e}")
        pass
    
    # print(f"Found POS groups for {dataset_name}: {ordered_groups}")
    # print(f"UPOS mapping: {upos_mapping}")
    
    return ordered_groups, upos_mapping

def get_pred_path(dataset, model, task, probe):
    """Find CSV file for a given model, checking both probes and probes2 directories."""
    # Handle different probe type naming conventions - try both "nn" and "mlp"
    probe_variants = [probe]
    if probe == "nn":
        probe_variants = ["nn", "mlp"]  # Try nn first, then mlp as fallback
    elif probe == "mlp":
        probe_variants = ["mlp", "nn"]  # Try mlp first, then nn as fallback
    
    # Check both probes and probes2 directories
    for probe_variant in probe_variants:
        probe_dirs = [
            os.path.join(OUTPUT_DIR, f"{dataset}_{model}_{task}_{probe_variant}"),
            os.path.join("../output/probes2", f"{dataset}_{model}_{task}_{probe_variant}")
        ]
        
        for probe_dir in probe_dirs:
            csv_path = os.path.join(probe_dir, "predictions.csv")
            if os.path.exists(csv_path):
                # if probe_variant != probe:
                #     print(f"Note: Using {probe_variant} probe as fallback for {probe} in {dataset}_{model}_{task}")
                return csv_path
    
    return None

def safe_read_csv(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_or_create_lemma_pos_mapping(conllu_path, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            # print(f"Loaded cached lemma-POS mapping with {len(cached_data)} entries")
            return cached_data

    lemma_to_pos = {}
    # print(f"Parsing {conllu_path} to create lemma-POS mapping...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 10:  # Ensure we have enough fields for CoNLL-U
                # Field 1 is the token ID, field 2 is FORM, field 3 is LEMMA, field 4 is UPOS
                token_id = parts[0]
                form = parts[1]
                lemma = parts[2] 
                upos = parts[3]
                
                # Skip multi-word tokens and empty nodes
                if '-' in token_id or '.' in token_id:
                    continue
                    
                if form and upos and form != "_" and upos != "_": 
                    # Store multiple variations to improve matching
                    lemma_to_pos[form.lower()] = upos
                    lemma_to_pos[form] = upos  # Keep original case
                if lemma and upos and lemma != "_" and upos != "_": 
                    lemma_to_pos[lemma.lower()] = upos
                    lemma_to_pos[lemma] = upos  # Keep original case
    
    # print(f"Created lemma-POS mapping with {len(lemma_to_pos)} entries")
    # print(f"Sample mappings: {dict(list(lemma_to_pos.items())[:10])}")
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_to_pos, f, ensure_ascii=False, indent=2)
    # print(f"Lemma-POS mapping cached to {cache_path}")
    return lemma_to_pos

def get_grouped_accuracy(df):
    if "y_true_str" not in df.columns or "y_pred_str" not in df.columns:
        return None, None

    counts = df.groupby("y_true_str").size()
    accs = df.groupby("y_true_str")["y_pred_str"].apply(
        lambda preds: np.mean(preds.values == df.loc[preds.index, "y_true_str"].values)
    )
    return accs, counts

def create_error_breakdown_table(dataset_name, model_list=None, probe_types=None):
    """
    Create error breakdown table for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "ud_gum_dataset")
        model_list: List of models to analyze (uses MODELS if None)
        probe_types: List of probe types (uses PROBE_TYPES if None)
    """
    if model_list is None:
        model_list = MODELS
    if probe_types is None:
        probe_types = PROBE_TYPES
    
    # Get POS information for this dataset
    ordered_pos_groups, upos_to_target_group = get_pos_info_for_dataset(dataset_name)
    
    # print(f"Creating error breakdown table for {dataset_name}")
    # print(f"Models: {model_list}")
    # print(f"Probe types: {probe_types}")
    # print(f"POS groups: {ordered_pos_groups}")
    
    # Get CoNLL-U path and cache path for this dataset
    conllu_path = get_conllu_path_from_dataset(dataset_name)
    pos_cache_path = get_pos_cache_path(dataset_name)
    lemma_pos_cache_path = get_lemma_pos_cache_path(dataset_name)
    
    if not conllu_path:
        # print(f"No CoNLL-U file found for {dataset_name}, skipping...")
        return
    
    results = []
    missing = []

    INFLECTION_DISPLAY = {
        "3rd_pers": "3rd person", "base": "Base", "comparative": "Comparative",
        "past": "Past", "plural": "Plural", "positive": "Positive",
        "singular": "Singular", "superlative": "Superlative"
    }
    PROBE_DISPLAY = {"reg": "MLP", "nn": "Linear"}

    # First, collect all unique inflection groups from the data
    all_inflection_groups = set()

    for model in model_list:  # desc="Models"
        for probe in probe_types:
            path = get_pred_path(dataset_name, model, "inflection", probe)
            df = safe_read_csv(path)
            if df is None:
                missing.append((model, probe, path))
                continue
            accs, counts = get_grouped_accuracy(df)
            if accs is None:
                missing.append((model, probe, path))
                continue
            
            # Collect all groups found in this model's data
            all_inflection_groups.update(accs.index)
            
            for grp, acc in accs.items():
                results.append({
                    "Model": model,
                    "Probe": probe,
                    "Group": grp,
                    "N": counts[grp],
                    "Accuracy": acc
                })

    # Create ordered list of inflection groups based on what we found and what we can display
    ordered_inflection_groups_internal = [k for k in all_inflection_groups if k in INFLECTION_DISPLAY]

    # Fill in missing combinations
    existing = {(r['Model'], r['Probe'], r['Group']) for r in results}
    for model in model_list:
        for probe in probe_types:
            for grp in ordered_inflection_groups_internal:
                if (model, probe, grp) not in existing:
                    results.append({
                        "Model": model,
                        "Probe": probe,
                        "Group": grp,
                        "N": 0,
                        "Accuracy": np.nan
                    })

    res_df = pd.DataFrame(results)

    # Create inflection table
    if ordered_inflection_groups_internal:
        inflection_cols = pd.MultiIndex.from_product(
            [[INFLECTION_DISPLAY[g] for g in ordered_inflection_groups_internal], [PROBE_DISPLAY[p] for p in probe_types]],
            names=["Inflection", "Probe"]
        )
        table_rows = []
        for model in model_list:
            row = []
            for grp in ordered_inflection_groups_internal:
                for pr in probe_types:
                    m = res_df[(res_df.Model == model) & (res_df.Probe == pr) & (res_df.Group == grp)]
                    val = m.Accuracy.iloc[0] if not m.empty else np.nan
                    row.append(f"{val:.3f}" if not np.isnan(val) else "--")
            table_rows.append(row)
        model_display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_list]
        inflection_df = pd.DataFrame(table_rows, index=model_display_names, columns=inflection_cols)
    else:
        inflection_df = pd.DataFrame()

    # Get inflection group counts
    canonical_inflection_df_for_counts = None
    preferred_probe_order_for_counts = ["reg", "nn"]

    if model_list and probe_types:
        for model_name_for_counts in model_list:
            for probe_type_for_counts in preferred_probe_order_for_counts:
                if probe_type_for_counts not in probe_types:
                    continue

                path_for_counts = get_pred_path(dataset_name, model_name_for_counts, "inflection", probe_type_for_counts)
                df_candidate_for_counts = safe_read_csv(path_for_counts)
                
                if df_candidate_for_counts is not None and \
                   "y_true_str" in df_candidate_for_counts.columns and \
                   not df_candidate_for_counts.empty:
                    
                    canonical_inflection_df_for_counts = df_candidate_for_counts
                    # print(f"INFO: Using '{path_for_counts}' for inflection group counts.")
                    break
            if canonical_inflection_df_for_counts is not None:
                break

    inflection_group_counts = {}
    if canonical_inflection_df_for_counts is not None:
        if 'layer' in canonical_inflection_df_for_counts.columns and not canonical_inflection_df_for_counts.empty:
            representative_layer = canonical_inflection_df_for_counts['layer'].min()
            single_layer_df = canonical_inflection_df_for_counts[canonical_inflection_df_for_counts['layer'] == representative_layer]
            # print(f"INFO: For inflection counts, using layer {representative_layer} from canonical file. Shape: {single_layer_df.shape}")
            counts_from_canonical_inflection = single_layer_df.groupby("y_true_str").size()
        else:
            # print("WARNING: 'layer' column not found in canonical inflection file or file empty after check. Using all data for counts.")
            counts_from_canonical_inflection = canonical_inflection_df_for_counts.groupby("y_true_str").size()
        
        for grp_key in ordered_inflection_groups_internal:
            display_name = INFLECTION_DISPLAY.get(grp_key)
            if display_name:
                count = counts_from_canonical_inflection.get(grp_key, 0)
                inflection_group_counts[display_name] = int(count)
    else:
        # print("WARNING: No canonical inflection predictions file found. Inflection group counts will be zero.")
        for grp_key in ordered_inflection_groups_internal:
            display_name = INFLECTION_DISPLAY.get(grp_key, grp_key)
            inflection_group_counts[display_name] = 0

    # Load lemma-POS mapping (using separate cache)
    lemma_pos_map = load_or_create_lemma_pos_mapping(conllu_path, lemma_pos_cache_path)

    # Find canonical lexeme file
    canonical = None
    for model in model_list:
        for pr in probe_types:
            df = safe_read_csv(get_pred_path(dataset_name, model, "lexeme", pr))
            if df is not None and "y_true_str" in df.columns:
                canonical = df.copy()
                break
        if canonical is not None:
            break

    pos_group_counts = {group: 0 for group in ordered_pos_groups}

    if canonical is not None:
        # Use data from a single representative layer for unique lexeme counts for POS grouping
        if 'layer' in canonical.columns and not canonical.empty:
            representative_layer_lex = canonical['layer'].min()
            canonical_single_layer = canonical[canonical['layer'] == representative_layer_lex]
            # print(f"INFO: For POS group counts, using layer {representative_layer_lex} from canonical lexeme file. Shape: {canonical_single_layer.shape}")
            unique_lexemes_in_dataset = canonical_single_layer.y_true_str.unique()
        else:
            # print("WARNING: 'layer' column not found in canonical lexeme file or file empty. Using all data for unique lexemes.")
            unique_lexemes_in_dataset = canonical.y_true_str.unique()

        # Debug: check what lexemes we have and their mappings
        total_lexemes = len(unique_lexemes_in_dataset)
        mapped_lexemes = 0
        
        # Try multiple matching strategies
        def get_pos_for_lexeme(lexeme, lemma_pos_map):
            """Try multiple strategies to find POS for a lexeme."""
            # Strategy 1: Direct match
            if lexeme in lemma_pos_map:
                return lemma_pos_map[lexeme]
            
            # Strategy 2: Lowercase match
            if lexeme.lower() in lemma_pos_map:
                return lemma_pos_map[lexeme.lower()]
                
            # Strategy 3: Try without common prefixes/suffixes for some languages
            # This is a simple heuristic and could be improved
            for prefix in ['не', 'un', 'in', 'de', 'dis']:  # Common negative prefixes
                if lexeme.lower().startswith(prefix) and len(lexeme) > len(prefix) + 2:
                    stem = lexeme.lower()[len(prefix):]
                    if stem in lemma_pos_map:
                        return lemma_pos_map[stem]
            
            return None
        
        # First pass: count lexemes by POS group
        temp_pos_counts = {group: 0 for group in ordered_pos_groups}
        
        for lexeme in unique_lexemes_in_dataset:
            upos = get_pos_for_lexeme(lexeme, lemma_pos_map)
            if upos:
                mapped_lexemes += 1
                pos_group = upos_to_target_group.get(upos, "Other")
            else:
                pos_group = "Other"
            
            if pos_group in temp_pos_counts:
                temp_pos_counts[pos_group] += 1
            else:
                temp_pos_counts["Other"] += 1
        
        # Second pass: consolidate small groups into "Other"
        # Groups with fewer than 10 lexemes will be moved to "Other"
        MIN_GROUP_SIZE = 10
        
        for group_name, count in temp_pos_counts.items():
            if group_name != "Other" and count < MIN_GROUP_SIZE:
                temp_pos_counts["Other"] += count
                temp_pos_counts[group_name] = 0
        
        # Create final pos_group_counts with only non-zero groups
        pos_group_counts = {group: count for group, count in temp_pos_counts.items() if count > 0}
        
        # Update ordered_pos_groups to only include groups with sufficient data
        ordered_pos_groups = [group for group in ordered_pos_groups if pos_group_counts.get(group, 0) > 0]
        
        # Create a modified mapping function that consolidates small groups
        def get_final_pos_group(lexeme, lemma_pos_map, min_size=MIN_GROUP_SIZE):
            """Get POS group, consolidating small groups into Other."""
            upos = get_pos_for_lexeme(lexeme, lemma_pos_map)
            if upos:
                pos_group = upos_to_target_group.get(upos, "Other")
                # If this group was too small, put it in Other
                if temp_pos_counts.get(pos_group, 0) < min_size and pos_group != "Other":
                    return "Other"
                return pos_group
            else:
                return "Other"
        
        pos_results = [] 
        for model in model_list:  # desc="Models (lexeme POS groups)"
            for pr in probe_types:
                df = safe_read_csv(get_pred_path(dataset_name, model, "lexeme", pr))
                if df is None or "y_true_str" not in df.columns or "y_pred_str" not in df.columns:
                    for pos_g in ordered_pos_groups: 
                        pos_results.append({
                            "Model": model,
                            "Probe": pr,
                            "POS_Group": pos_g, 
                            "Accuracy": np.nan
                        })
                    continue
                
                # Use the improved matching function that consolidates small groups
                df['pos_group'] = df.y_true_str.apply(
                    lambda lem: get_final_pos_group(lem, lemma_pos_map)
                )

                for pos_g in ordered_pos_groups: 
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
            [ordered_pos_groups, [PROBE_DISPLAY[p] for p in probe_types]],
            names=["Part of Speech", "Probe"] 
        )
        lex_rows_for_table = []
        for model_id in model_list: 
            row = []
            for pos_g in ordered_pos_groups: 
                for pr_type in probe_types:
                    m = lex_df_data[(lex_df_data.Model == model_id) & (lex_df_data.Probe == pr_type) & (lex_df_data.POS_Group == pos_g)]
                    val = m.Accuracy.values[0] if not m.empty and not pd.isna(m.Accuracy.values[0]) else np.nan
                    row.append(f"{val:.3f}" if not np.isnan(val) else "--")
            lex_rows_for_table.append(row)
        
        model_display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_list]
        lexeme_df = pd.DataFrame(lex_rows_for_table, index=model_display_names, columns=lexeme_cols_for_table)
    else:
        empty_cols = pd.MultiIndex.from_product(
            [ordered_pos_groups, [PROBE_DISPLAY[p] for p in probe_types]],
            names=["Part of Speech", "Probe"]
        )
        lexeme_df = pd.DataFrame(columns=empty_cols)

    # Generate LaTeX tables
    def to_latex_combined_probe_table(df, caption_text, label, group_counts_map=None):
        if df.columns.empty:
            return f"% No data columns available for table: {caption_text}\n% DataFrame had no columns.\n"

        # Get the columns for each probe type
        try:
            lr_df = df.xs("MLP", axis=1, level="Probe")
            mlp_df = df.xs("Linear", axis=1, level="Probe")
        except KeyError:
            return f"% Could not extract probe data for combined table: {caption_text}\n"

        # Determine number of columns for each probe type
        n_cols_per_probe = lr_df.shape[1]
        total_cols = n_cols_per_probe * 2

        latex = (
            '\\begin{table*}[t]\n'
            '\\small\n'
            '\\centering\n'
            '\\renewcommand\\arraystretch{1.2}\n'
            '\\setlength{\\tabcolsep}{3pt}\n' 
            '\\begin{minipage}{\\linewidth}\\centering\n'
            f'\\begin{{tabular}}{{@{{}}l{"c" * total_cols}@{{}}}}\n'
            '    \\toprule\n'
        )
        
        # Header row 1: Model and probe type groupings
        header_row1 = f'    \\multirow{{3}}{{*}}{{\\textbf{{Model}}}} & \\multicolumn{{{n_cols_per_probe}}}{{c}}{{\\textbf{{Linear Regression}}}} & \\multicolumn{{{n_cols_per_probe}}}{{c}}{{\\textbf{{MLP}}}} \\\\\n'
        latex += header_row1
        
        # Header row 2: cmidrules and column names
        cmidrule_start_lr = 2
        cmidrule_end_lr = 1 + n_cols_per_probe
        cmidrule_start_mlp = cmidrule_end_lr + 1
        cmidrule_end_mlp = cmidrule_start_mlp + n_cols_per_probe - 1
        
        header_row2 = f'    \\cmidrule(lr){{{cmidrule_start_lr}-{cmidrule_end_lr}}} \\cmidrule(lr){{{cmidrule_start_mlp}-{cmidrule_end_mlp}}}\n'
        latex += header_row2
        
        # Header row 3: column names for each group
        header_row3 = '    '
        for col_name in lr_df.columns:
            header_row3 += f' & {col_name}'
        for col_name in mlp_df.columns:
            header_row3 += f' & {col_name}'
        header_row3 += ' \\\\\n'
        latex += header_row3

        # Header row 4: counts (if provided)
        if group_counts_map:
            header_row4 = '    '
            # Counts for Linear Regression columns
            for col_name in lr_df.columns:
                count = group_counts_map.get(str(col_name), "--") 
                n_str = f"{count:,}" if isinstance(count, int) else str(count)
                header_row4 += f" & \\footnotesize (n={n_str})"
            # Same counts for MLP columns (since they're the same groups)
            for col_name in mlp_df.columns:
                count = group_counts_map.get(str(col_name), "--") 
                n_str = f"{count:,}" if isinstance(count, int) else str(count)
                header_row4 += f" & \\footnotesize (n={n_str})"
            header_row4 += " \\\\\n"
            latex += header_row4
            
        latex += "    \\midrule\n"

        # Data rows
        for idx, (lr_row, mlp_row) in zip(lr_df.index, zip(lr_df.values, mlp_df.values)):
            latex += f"    {idx}"
            # Add Linear Regression values
            for val in lr_row:
                latex += f" & {val}"
            # Add MLP values
            for val in mlp_row:
                latex += f" & {val}"
            latex += " \\\\\n"
            
        latex += "    \\bottomrule\n"
        latex += "\\end{tabular}\n"
        
        caption_suffix = (
            " Accuracies are calculated over all examples for a given group across all layers. "
            "Counts (n) are derived from a single representative layer for each group. "
            "All accuracy values are on a 0--1 scale."
        )
        latex += f"\\caption{{{caption_text}{caption_suffix}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\end{minipage}\n"
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
            "All accuracy values are on a 0--1 scale."
        )
        latex += f"\\caption{{{caption_text}{caption_suffix}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\end{minipage}\n"
        latex += "\\end{table*}\n"
        return latex

    # Define which tables should NOT be combined
    skip_combined_tables = {
        ("ud_zh_gsd_dataset", "lexeme"),
        ("ud_de_gsd_dataset", "inflection"), 
        ("ud_fr_gsd_dataset", "inflection"),
        ("ud_ru_syntagrus_dataset", "inflection"),
        ("ud_tr_imst_dataset", "inflection")
    }

    # Print LaTeX tables
    if not inflection_df.empty:
        language_name = get_language_name_from_dataset(dataset_name)
        
        if (dataset_name, "inflection") in skip_combined_tables:
            # Generate separate tables
            inflection_caption_lr = (
                f"Breakdown of inflection classification accuracy for each model by inflection type using Linear Regression classifiers ({language_name})."
            )
            inflection_caption_mlp = (
                f"Breakdown of inflection classification accuracy for each model by inflection type using Multi-Layer Perceptron (MLP) classifiers ({language_name})."
            )

            print(to_latex_single_probe_table(inflection_df, "reg", inflection_caption_lr, f"tab:inflection_breakdown_lr_{dataset_name}", inflection_group_counts))
            print(to_latex_single_probe_table(inflection_df, "nn", inflection_caption_mlp, f"tab:inflection_breakdown_mlp_{dataset_name}", inflection_group_counts))
        else:
            # Generate combined table
            inflection_caption_combined = (
                f"Breakdown of inflection classification accuracy for each model by inflection type using Linear Regression and Multi-Layer Perceptron (MLP) classifiers ({language_name})."
            )
            print(to_latex_combined_probe_table(inflection_df, inflection_caption_combined, f"tab:inflection_breakdown_combined_{dataset_name}", inflection_group_counts))

    if not lexeme_df.empty:
        language_name = get_language_name_from_dataset(dataset_name)
        
        if (dataset_name, "lexeme") in skip_combined_tables:
            # Generate separate tables
            lexeme_caption_pos_lr = ( 
                f"Breakdown of lexeme classification accuracy by Part of Speech (POS) for each model, using Linear Regression classifiers ({language_name}). "
                "Lexemes are grouped by their POS tags (e.g., Noun, Verb, Adjective)."
            )
            lexeme_caption_pos_mlp = ( 
                f"Breakdown of lexeme classification accuracy by Part of Speech (POS) for each model, using Multi-Layer Perceptron (MLP) classifiers ({language_name}). "
                "Lexemes are grouped by their POS tags (e.g., Noun, Verb, Adjective)."
            )

            print(to_latex_single_probe_table(lexeme_df, "reg", lexeme_caption_pos_lr, f"tab:lexeme_pos_breakdown_lr_{dataset_name}", pos_group_counts))
            print(to_latex_single_probe_table(lexeme_df, "nn", lexeme_caption_pos_mlp, f"tab:lexeme_pos_breakdown_mlp_{dataset_name}", pos_group_counts))
        else:
            # Generate combined table
            lexeme_caption_pos_combined = (
                f"Breakdown of lexeme classification accuracy by Part of Speech (POS) for each model, using Linear Regression and Multi-Layer Perceptron (MLP) classifiers ({language_name}). "
                "Lexemes are grouped by their POS tags (e.g., Noun, Verb, Adjective)."
            )
            print(to_latex_combined_probe_table(lexeme_df, lexeme_caption_pos_combined, f"tab:lexeme_pos_breakdown_combined_{dataset_name}", pos_group_counts))

def get_models_for_dataset(dataset_name):
    """
    Get the appropriate models for each dataset based on what was actually tested.
    """
    if dataset_name == "ud_gum_dataset":
        # English dataset - all models were tested
        return MODELS
    elif dataset_name == "ud_zh_gsd_dataset":
        # Chinese dataset - only multilingual models and Chinese goldfish
        return ["mt5", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct", "goldfish_zho_hans_1000mb"]
    elif dataset_name == "ud_de_gsd_dataset":
        # German dataset - only multilingual models and German goldfish
        return ["mt5", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct", "goldfish_deu_latn_1000mb"]
    elif dataset_name == "ud_fr_gsd_dataset":
        # French dataset - only multilingual models and French goldfish
        return ["mt5", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct", "goldfish_fra_latn_1000mb"]
    elif dataset_name == "ud_ru_syntagrus_dataset":
        # Russian dataset - only multilingual models and Russian goldfish
        return ["mt5", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct", "goldfish_rus_cyrl_1000mb"]
    elif dataset_name == "ud_tr_imst_dataset":
        # Turkish dataset - only multilingual models and Turkish goldfish
        return ["mt5", "qwen2", "qwen2-instruct", "qwen2.5-7B", "qwen2.5-7B-instruct", "goldfish_tur_latn_1000mb"]
    else:
        # Fallback to all models
        return MODELS

# Add a main function to process all datasets
def main():
    """Process all available datasets."""
    all_datasets = [
        # "ud_gum_dataset",
        "ud_zh_gsd_dataset",
        "ud_de_gsd_dataset",
        "ud_fr_gsd_dataset",
        "ud_ru_syntagrus_dataset",
        "ud_tr_imst_dataset",
    ]
    
    for dataset in all_datasets:
        # print(f"\n{'='*60}")
        # print(f"Processing dataset: {dataset}")
        # print('='*60)
        
        # Get the appropriate models for this dataset
        dataset_models = get_models_for_dataset(dataset)
        create_error_breakdown_table(dataset, model_list=dataset_models)

if __name__ == "__main__":
    main()