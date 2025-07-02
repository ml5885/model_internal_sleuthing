import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src import config

def load_shards(path):
    if os.path.isdir(path):
        files = sorted(
            [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith(".npz") and "activations_part" in f],
            key=lambda fn: int(re.search(r"part_?(\d+)", os.path.basename(fn)).group(1))
        )
        if not files:
            raise ValueError(f"No activation shards found in directory: {path}")
        return files
    if os.path.isfile(path) and path.endswith(".npz"):
        return [path]
    raise ValueError(f"{path} is not a .npz file or directory of shards")

def load_layer(shards, layer_idx):
    parts = []
    for shard in shards:
        try:
            with np.load(shard, mmap_mode="r") as data:
                arr = data["activations"]
        except EOFError:
            raise RuntimeError(f"Corrupt or empty shard: {shard}")
        except Exception:
            with np.load(shard) as data:
                arr = data["activations"]
        parts.append(arr[:, layer_idx, :])
    return np.concatenate(parts, axis=0)

def run_inflection_steering(activations_dir, probe_objs, pred_dfs, valid_mask, 
                            keep_mask, inf_labels, layers, lambdas, outdir):
    rows = []
    for layer in layers:
        if layer not in pred_dfs or pred_dfs[layer].empty:
            print(f"Warning: Layer {layer} has no prediction data. Skipping steering.")
            continue

        shards = load_shards(activations_dir)
        X = load_layer(shards, layer)
        X_full = X[valid_mask]
        y_full = inf_labels
        unique_inflections, inf_counts = np.unique(y_full, return_counts=True)
        # Only keep inflections with >=10 samples
        valid_inflections = unique_inflections[inf_counts >= 10]
        inf_mask = np.isin(y_full, valid_inflections)
        X_valid = X_full[inf_mask]
        y_valid = y_full[inf_mask]

        # Compute AIVs for each inflection
        aiv_dict = {}
        for infl in valid_inflections:
            aiv_dict[infl] = X_valid[y_valid == infl].mean(0)

        # Choose 100 random test words (rows)
        rng = np.random.RandomState(config.SEED)
        test_indices = rng.choice(np.arange(len(X_valid)), size=min(100, len(X_valid)), replace=False)
        H_test = X_valid[test_indices]
        y_test = y_valid[test_indices]

        obj = probe_objs[layer]
        # Get baseline predictions and probabilities
        if hasattr(obj, "predict_proba"):
            y_probs_base = obj.predict_proba(H_test)
            y_pred_base = obj.predict(H_test)
        elif hasattr(obj, "predict"):
            y_logits_base = obj.predict(H_test, batch_size=config.TRAIN_PARAMS["batch_size"])
            y_probs_base = np.exp(y_logits_base) / np.exp(y_logits_base).sum(axis=1, keepdims=True)
            y_pred_base = y_logits_base.argmax(1)
        else:  # ridge regression
            y_logits_base = H_test.dot(obj)
            y_probs_base = np.exp(y_logits_base) / np.exp(y_logits_base).sum(axis=1, keepdims=True)
            y_pred_base = y_logits_base.argmax(1)

        for lam in lambdas:
            total_flips = 0
            total_prob_change = 0
            total_trials = 0
            for i in range(len(H_test)):
                h = H_test[i]
                infl_true = y_test[i]
                # For each other inflection (not the target's inflection)
                for infl_other in valid_inflections:
                    if infl_other == infl_true:
                        continue
                    # Steering vector: aiv[target] - lambda * aiv[other]
                    steering_vec = aiv_dict[infl_true] - lam * (aiv_dict[infl_other])
                    h_steered = h + steering_vec
                    # Get steered predictions and probabilities
                    if hasattr(obj, "predict_proba"):
                        y_probs_steered = obj.predict_proba(h_steered[np.newaxis, :])[0]
                        y_pred_steered = obj.predict(h_steered[np.newaxis, :])[0]
                    elif hasattr(obj, "predict"):
                        y_logits_steered = obj.predict(h_steered[np.newaxis, :], batch_size=1)[0]
                        y_probs_steered = np.exp(y_logits_steered) / np.exp(y_logits_steered).sum()
                        y_pred_steered = y_logits_steered.argmax()
                    else:
                        y_logits_steered = h_steered.dot(obj)
                        y_probs_steered = np.exp(y_logits_steered) / np.exp(y_logits_steered).sum()
                        y_pred_steered = y_logits_steered.argmax()
                    # Track probability change for steered inflection
                    prob_change = y_probs_steered[infl_other] - y_probs_base[i][infl_other]
                    total_prob_change += prob_change
                    # Track flip rate
                    if y_pred_steered == infl_other:
                        total_flips += 1
                    total_trials += 1
            flip_rate = total_flips / total_trials if total_trials > 0 else np.nan
            avg_prob_delta = total_prob_change / total_trials if total_trials > 0 else np.nan
            rows.append({
                "layer": layer,
                "lambda": lam,
                "flip_rate": flip_rate,
                "prob_delta": avg_prob_delta
            })

        # Plot both metrics for this layer
        df_layer = pd.DataFrame([r for r in rows if r["layer"] == layer])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(df_layer["lambda"], df_layer["flip_rate"], marker="o", color='blue')
        ax1.set_xlabel("lambda")
        ax1.set_ylabel("Flip Rate")
        ax1.set_title(f"Layer {layer} Steering Flip Rate")
        ax1.grid(True, alpha=0.3)
        ax2.plot(df_layer["lambda"], df_layer["prob_delta"], marker="s", color='red')
        ax2.set_xlabel("lambda")
        ax2.set_ylabel("Average Probability Delta")
        ax2.set_title(f"Layer {layer} Target Inflection Probability Change")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"steering_metrics_layer{layer}.png"))
        plt.close()

    # Save CSV + print Markdown pivot tables
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "steering_results.csv")
    df.to_csv(csv_path, index=False)

    if not df.empty:
        print("## Flip Rate Results:")
        print(
            df.pivot_table(index="layer", columns="lambda", values="flip_rate")
              .to_markdown(floatfmt=".3f")
        )
        print("\n## Probability Delta Results:")
        print(
            df.pivot_table(index="layer", columns="lambda", values="prob_delta")
              .to_markdown(floatfmt=".3f")
        )
    else:
        print("No steering results to display; the results DataFrame is empty.")
