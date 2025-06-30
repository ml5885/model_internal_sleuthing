import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "../data"
DATASETS = [
    "ud_gum_dataset",
    # "ud_ru_syntagrus_dataset", "ud_tr_imst_dataset",
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
    return labels

def get_control_labels(df):
    if "Word Form" not in df.columns: return None
    return df["Word Form"].astype('category').cat.codes.values

def analyze_splits(y_true):
    if len(y_true) == 0:
        return "no rows to split"
    try:
        _, y_tmp = train_test_split(
            np.zeros(len(y_true)), y_true,
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
            y_true = get_labels(df, task)
            y_control = get_control_labels(df)
            if y_true is None:
                print("  Task labels not available, skipping.")
                continue
            true_counts = np.bincount(y_true)
            control_counts = np.bincount(y_control)
            for threshold in THRESHOLDS:
                print(f"\n  [Filtering with Threshold >= {threshold}]")
                keep_mask_true = true_counts[y_true] >= threshold
                keep_mask_control = control_counts[y_control] >= threshold
                final_mask = keep_mask_true & keep_mask_control
                n_kept_rows = np.sum(final_mask)
                y_filtered = y_true[final_mask]
                n_kept_classes = len(np.unique(y_filtered))
                print(f"    - Rows remaining: {n_kept_rows} / {len(df)} ({n_kept_rows/len(df):.1%})")
                print(f"    - Classes remaining: {n_kept_classes} / {len(true_counts)}")
                split_status = analyze_splits(y_filtered)
                print(f"    - Splitting status: {split_status}")
                if "failed" in split_status or "singleton" in split_status or "empty" in split_status:
                    print("      [WARNING] This configuration may cause issues during probe training.")
    print(f"\n{'='*20} Analysis Complete {'='*20}\n")

if __name__ == "__main__":
    main()
