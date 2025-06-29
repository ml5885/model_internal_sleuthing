#!/usr/bin/env python
import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = [
    "ud_gum_dataset", "ud_zh_gsd_dataset", "ud_de_gsd_dataset",
    "ud_fr_gsd_dataset", "ud_ru_syntagrus_dataset", "ud_tr_imst_dataset",
]
TASKS = ["lexeme", "inflection"]
THRESHOLDS = [3, 4, 5]
OUTPUT_DIR = "output"
SEED = 42

def make_labels(df, task):
    lex_lookup = {w: i for i, w in enumerate(sorted(set(df["Lemma"])))}
    lex_labels = np.array([lex_lookup[w] for w in df["Lemma"]], dtype=int)
    if "Inflection Label" in df.columns:
        infl_lookup = {x: i for i, x in enumerate(sorted(set(df["Inflection Label"])))}
        infl_labels = np.array([infl_lookup[x] for x in df["Inflection Label"]], dtype=int)
    else:
        infl_labels = lex_labels
    return lex_labels if task == "lexeme" else infl_labels

def rows_kept_mask(df, y_true, threshold):
    wf_lookup = {w: i for i, w in enumerate(sorted(set(df["Word Form"])))}
    y_ctrl = np.array([wf_lookup[w] for w in df["Word Form"]], dtype=int)
    keep_true = np.bincount(y_true)[y_true] >= threshold
    keep_ctrl = np.bincount(y_ctrl)[y_ctrl] >= threshold
    return keep_true & keep_ctrl

def test_rows_after_split(y_true, rows):
    X_dummy = np.empty((rows, 1), dtype=np.float32)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_dummy, y_true, train_size=0.70, random_state=SEED, stratify=y_true
    )
    val_frac = 0.3333333333333333
    strat_val = y_tmp if np.bincount(y_tmp).min() > 1 else None
    _, X_test, _, _ = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, random_state=SEED, stratify=strat_val
    )
    return len(X_test)

def report(df, label, task):
    y_true = make_labels(df, task)
    for thr in THRESHOLDS:
        mask = rows_kept_mask(df, y_true, thr)
        kept_rows = int(mask.sum())
        print(f"  {label} | ≥{thr}: kept {kept_rows:6d}", end="")
        if kept_rows == 0:
            print("  (no rows)")
            continue
        try:
            n_test = test_rows_after_split(y_true[mask], kept_rows)
            note = "" if n_test else " (!!) test empty – layer skipped"
            print(f"  → test {n_test:6d}{note}")
        except ValueError as e:
            print(f"  → split error: {e.args[0]!s}  ⇒ layer skipped")

def main():
    for ds in DATASETS:
        csv_path = os.path.join("data", f"{ds}.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] missing {csv_path}")
            continue
        raw = pd.read_csv(csv_path)
        base_mask = raw["Lemma"].notna()
        if "Inflection Label" in raw.columns:
            base_mask &= raw["Inflection Label"].notna()
        raw = raw[base_mask].reset_index(drop=True)
        print(f"\n=== {ds} ===")
        for task in TASKS:
            print(f"\n-- {task.upper()} --")
            report(raw, "FULL CSV", task)
            pat = os.path.join(OUTPUT_DIR, "**", f"*{ds}_reps", "sampled_indices.csv")
            for sp in sorted(glob.glob(pat, recursive=True)):
                idx = pd.read_csv(sp)["index"].values
                if idx.max() >= len(raw):
                    continue
                sub = raw.iloc[idx].reset_index(drop=True)
                report(sub, os.path.relpath(sp), task)
    print("\nDone.")

if __name__ == "__main__":
    main()
