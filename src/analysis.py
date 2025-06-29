import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm
from src import config, utils

def list_shards(path_or_dir):
    """
    Return a sorted list of .npz shard filepaths under path_or_dir.
    """
    files = [
        f for f in os.listdir(path_or_dir)
        if f.endswith('.npz') and 'activations_part' in f
    ]
    def idx(fname):
        m = re.search(r'part_?(\d+)', fname)
        return int(m.group(1)) if m else -1
    files.sort(key=idx)
    return [os.path.join(path_or_dir, f) for f in files]

def shard_loader(shard_path, layer_idx):
    """
    Load only the activations for a single layer from a .npz shard.
    """
    arr = np.load(shard_path, mmap_mode='r')
    X = arr['activations']    # shape (batch_size, n_layers, d_model)
    return X[:, layer_idx, :]  # shape (batch_size, d_model)

def avg_pairwise_cosine_stream(normed_shard_list, labels):
    """
    Compute avg pairwise cosine similarity per label.
    """
    uniq = np.unique(labels)
    d_model = normed_shard_list[0].shape[1]
    sums = np.zeros((uniq.size, d_model), dtype=np.float64)
    counts = np.zeros(uniq.size, dtype=np.int64)

    offset = 0
    for Xn in tqdm(normed_shard_list, desc="  computing avg cosine", leave=False):
        B = Xn.shape[0]
        slice_labels = labels[offset:offset+B]
        offset += B

        for lbl in uniq:
            mask = (slice_labels == lbl)
            if not mask.any():
                continue
            sums[lbl]  += Xn[mask].sum(axis=0)
            counts[lbl] += int(mask.sum())

    num = ((np.linalg.norm(sums, axis=1)**2 - counts) / 2).sum()
    denom = ((counts * (counts - 1)) / 2).sum()
    return float(num / denom) if denom > 0 else 0.0

def unsupervised_layer_analysis(activations_input, labels_file, model_key, dataset_key):
    # Gather shard files
    shards = list_shards(activations_input)
    if not shards:
        raise ValueError(f"No .npz shards found in {activations_input}")

    # Peek at first shard to get dimensions
    sample = np.load(shards[0], mmap_mode='r')['activations']
    _, n_layers, d_model = sample.shape
    batch_size = sample.shape[0]

    # Load labels
    df = pd.read_csv(labels_file)
    
    # If activations were sampled, filter labels to match
    sampled_indices_path = os.path.join(activations_input, "sampled_indices.csv")
    if os.path.exists(sampled_indices_path):
        sampled_df = pd.read_csv(sampled_indices_path)
        original_indices = sampled_df['index'].values
        df = df.iloc[original_indices].reset_index(drop=True)
        utils.log_info(f"Loaded {len(df)} labels corresponding to sampled activations.")

    inf_labels = pd.Categorical(df["Inflection Label"]).codes
    lex_labels = pd.Categorical(df["Lemma"]).codes
    lex_categories = pd.Categorical(df["Lemma"]).categories

    # Prepare output directory
    outdir = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset_key}_analysis")
    os.makedirs(outdir, exist_ok=True)

    # Find unique sentences for clustering
    unique_sentences = df["Sentence"].values
    _, unique_idx = np.unique(unique_sentences, return_index=True)
    unique_idx = np.sort(unique_idx)
    n_unique = unique_idx.size
    cluster_matrix = np.zeros((n_unique, n_layers), dtype=int)

    records = []

    # Process one layer at a time
    for layer in tqdm(range(n_layers), desc="Analyzing Layers", dynamic_ncols=True):
        # 1) Build a list of normalized shards for this layer
        normed_shards = []
        offset = 0
        for shard_path in tqdm(shards, desc=f"  layer {layer} normalize", leave=False):
            X = shard_loader(shard_path, layer)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xn = X / norms
            normed_shards.append(Xn)
            offset += Xn.shape[0]

        # 2) Average pairwise cosine
        cos_inf = avg_pairwise_cosine_stream(normed_shards, inf_labels)
        cos_lex = avg_pairwise_cosine_stream(normed_shards, lex_labels)
        records.append({
            "layer": layer,
            "cosine_inflection": cos_inf,
            "cosine_lexeme": cos_lex
        })

        # 3) Extract just the unique-index rows for this layer
        data_u = np.zeros((n_unique, d_model), dtype=np.float32)
        offset = 0
        for Xn in tqdm(normed_shards, desc=f"  layer {layer} unique extract", leave=False):
            B = Xn.shape[0]
            mask = (unique_idx >= offset) & (unique_idx < offset + B)
            if mask.any():
                local_pos = unique_idx[mask] - offset
                data_u[mask] = Xn[local_pos]
            offset += B

        # 4) PCA + t-SNE + KMeans
        pca = IncrementalPCA(
            n_components=min(len(np.unique(inf_labels)), n_unique),
            batch_size=batch_size
        )
        pca.fit(data_u)
        pca_proj = pca.transform(data_u)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_proj = tsne.fit_transform(pca_proj)

        clusters = KMeans(
            n_clusters=len(np.unique(inf_labels)),
            random_state=config.CLUSTERING["random_state"]
        ).fit_predict(tsne_proj)
        cluster_matrix[:, layer] = clusters

        # 5) Save t-SNE plots
        cmap = plt.get_cmap("tab10")
        colors_inf = [cmap(inf_labels[i] % 10) for i in unique_idx]
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors_inf, s=10)
        handles_inf = [
            plt.Line2D([0], [0], marker='o', color='w', label=lab,
                       markerfacecolor=cmap(i % 10), markersize=6)
            for i, lab in enumerate(pd.Categorical(df["Inflection Label"]).categories)
        ]
        plt.legend(handles=handles_inf, title="Inflection", fontsize='small', markerscale=0.7)
        plt.title(f"Layer {layer} t-SNE by Inflection")
        plt.savefig(os.path.join(outdir, f"layer_{layer}_tsne_inflection.png"), bbox_inches="tight")
        plt.close()

        # Lexeme t-SNE plot without legend (too many categories)
        colors_lex = [cmap(lex_labels[i] % 10) for i in unique_idx]
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors_lex, s=10)
        plt.title(f"Layer {layer} t-SNE by Lexeme (legend omitted)")
        plt.savefig(os.path.join(outdir, f"layer_{layer}_tsne_lexeme.png"), bbox_inches="tight")
        plt.close()

    # Save cosine-by-layer table
    cos_df = pd.DataFrame(records)
    cos_df.to_csv(os.path.join(outdir, "cosine_by_layer.csv"), index=False)

    # Plot overall cosine trends
    plt.figure(figsize=(6, 4))
    plt.plot(cos_df["layer"], cos_df["cosine_inflection"], marker="o", label="Inflection")
    plt.plot(cos_df["layer"], cos_df["cosine_lexeme"],   marker="s", label="Lexeme")
    plt.xlabel("Layer")
    plt.ylabel("Avg Pairwise Cosine")
    plt.title("Avg Pairwise Cosine by Layer")
    plt.legend()
    plt.savefig(os.path.join(outdir, "cosine_similarity_over_layers.png"), bbox_inches="tight")
    plt.close()

    # Save per-layer clusters for each unique sentence
    rows = []
    sentences = df["Sentence"].values
    for i, idx in enumerate(unique_idx):
        row = {
            "Index": int(idx),
            "Sentence": sentences[idx],
            "Inflection_Label": df.loc[idx, "Inflection Label"]
        }
        for lyr in range(n_layers):
            row[f"Cluster_L{lyr}"] = int(cluster_matrix[i, lyr])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "clusters_per_layer.csv"), index=False)

    # Save compressed analysis results
    np.savez_compressed(
        os.path.join(outdir, "analysis_results.npz"),
        results={
            r["layer"]: {
                "cosine_inflection": r["cosine_inflection"],
                "cosine_lexeme": r["cosine_lexeme"]
            }
            for r in records
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    unsupervised_layer_analysis(
        args.activations_dir,
        args.labels,
        args.model,
        args.dataset
    )
