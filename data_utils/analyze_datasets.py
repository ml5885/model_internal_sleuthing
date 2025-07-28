import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "../data"
DATASETS = [
    "ud_gum_dataset",
    "ud_ru_syntagrus_dataset", 
    "ud_tr_imst_dataset",
    "ud_zh_gsd_dataset",
    "ud_de_gsd_dataset",
    "ud_fr_gsd_dataset",
]
TASKS = ["lexeme", "inflection"]
THRESHOLDS = [2, 3, 4, 5]
SEED = 42
SPLIT_CONFIG = {"train_size": 0.7, "test_size": 0.2, "val_size": 0.1}

def get_labels(df, task):
    if task == "lexeme":
        if "Lemma" not in df.columns: return None
        labels = df["Lemma"].astype('category').cat.codes.values
    elif task == "inflection":
        if "Inflection Label" not in df.columns: return None
        labels = df["Inflection Label"].astype('category').cat.codes.values
    else:
        raise ValueError(f"Unknown task: {task}")
    # Remove negative codes (which represent NaN/missing)
    labels = labels[labels >= 0]
    return labels

def get_control_labels(df):
    if "Word Form" not in df.columns: return None
    labels = df["Word Form"].astype('category').cat.codes.values
    labels = labels[labels >= 0]
    return labels

def analyze_splits(y_true):
    if len(y_true) == 0:
        return "no rows to split"
    try:
        y_train, y_tmp = train_test_split(
            y_true,
            train_size=SPLIT_CONFIG["train_size"],
            random_state=SEED,
            stratify=y_true
        )
    except ValueError as e:
        return f"train/test split failed: {e}"
    if len(y_tmp) == 0:
        return "train split resulted in empty temp set"
    val_counts = np.bincount(y_tmp)
    if np.any(val_counts[val_counts > 0] == 1):
        return "val/test set has singleton classes, cannot stratify further"
    try:
        val_frac = SPLIT_CONFIG["val_size"] / (SPLIT_CONFIG["val_size"] + SPLIT_CONFIG["test_size"])
        y_val, y_test = train_test_split(
            y_tmp,
            train_size=val_frac,
            random_state=SEED,
            stratify=y_tmp
        )
        if len(y_test) == 0:
            return "test set is empty"
        return f"test set has {len(y_test)} samples"
    except ValueError as e:
        return f"val/test split failed: {e}"

def main():
    for dataset_name in DATASETS:
        csv_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            print(f"\n--- [SKIPPING] {dataset_name}: File not found at {csv_path} ---\n")
            continue
        print(f"\n{'='*20} Analyzing: {dataset_name} {'='*20}")
        df = pd.read_csv(csv_path)
        print("\n[Overall Statistics]")
        print(f"  - Total rows: {len(df)}")
        for col in ["Lemma", "Inflection Label", "Word Form"]:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"  - Unique '{col}' values: {unique_count}")
            else:
                print(f"  - Column '{col}' not found.")
        for task in TASKS:
            print(f"\n--- Task: {task.upper()} ---")
            # Determine which columns are needed for this task
            if task == "lexeme":
                task_col = "Lemma"
            elif task == "inflection":
                task_col = "Inflection Label"
            else:
                print("  Unknown task, skipping.")
                continue
            # Only keep rows where both task_col and "Word Form" are not null
            if task_col not in df.columns or "Word Form" not in df.columns:
                print("  Task or control labels not available, skipping.")
                continue
            nonnull_mask = df[task_col].notnull() & df["Word Form"].notnull()
            df_valid = df[nonnull_mask]
            if len(df_valid) == 0:
                print("  No valid rows after filtering missing values, skipping.")
                continue
            y_true = get_labels(df_valid, task)
            y_control = get_control_labels(df_valid)
            if y_true is None or y_control is None:
                print("  Task or control labels not available, skipping.")
                continue
            # Defensive: skip if empty after filtering
            if len(y_true) == 0 or len(y_control) == 0:
                print("  No valid rows after filtering negative codes, skipping.")
                continue
            true_counts = np.bincount(y_true)
            control_counts = np.bincount(y_control)
            print("  Threshold | Rows kept | % kept | Classes kept | Split status")
            for threshold in THRESHOLDS:
                keep_mask_true = true_counts[y_true] >= threshold
                keep_mask_control = control_counts[y_control] >= threshold
                final_mask = keep_mask_true & keep_mask_control
                n_kept_rows = np.sum(final_mask)
                y_filtered = y_true[final_mask]
                n_kept_classes = len(np.unique(y_filtered))
                pct_kept = n_kept_rows / len(df) * 100
                split_status = analyze_splits(y_filtered)
                warn = ""
                if "failed" in split_status or "singleton" in split_status or "empty" in split_status:
                    warn = " [!]"
                print(f"    {threshold:>5}    | {n_kept_rows:>8} | {pct_kept:6.1f}% | {n_kept_classes:>12} | {split_status}{warn}")
    print(f"\n{'='*20} Analysis Complete {'='*20}\n")

if __name__ == "__main__":
    main()
