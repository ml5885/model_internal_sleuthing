import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm
from src import config, utils

def load_activations(path_or_dir):
    if os.path.isdir(path_or_dir):
        files = sorted(
            [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir)
             if f.endswith('.npz') and 'activations_part' in f]
        )
        arrays = [np.load(f)['activations'] for f in files]
        return np.concatenate(arrays, axis=0)
    else:
        return np.load(path_or_dir)['activations']

def avg_group_cosine(acts: np.ndarray, labels: np.ndarray) -> float:
    total, count = 0.0, 0
    acts_norm = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    for lab in np.unique(labels):
        idxs = np.where(labels == lab)[0]
        k = idxs.size
        if k < 2:
            continue
        sub = acts_norm[idxs]
        sim = sub @ sub.T
        total += sim.sum() - np.trace(sim)
        count += k * (k - 1)
    return total / count if count > 0 else 0.0

def unsupervised_layer_analysis(activations_input, labels_file, model_key, dataset_key):
    # Figure out where our shards live
    if os.path.isdir(activations_input):
        shard_files = sorted([
            os.path.join(activations_input, f) for f in os.listdir(activations_input)
            if f.endswith('.npz') and 'activations_part' in f
        ])
    else:
        shard_files = [activations_input]

    # Load one shard just to get dims
    sample = np.load(shard_files[0])['activations']
    _, n_layers, d_model = sample.shape

    # Read labels & compute unique sentences index
    df = pd.read_csv(labels_file)
    sentences = df["Sentence"].values
    _, unique_idx = np.unique(sentences, return_index=True)
    unique_idx = np.sort(unique_idx)

    # Prepare output directory
    outdir = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset_key}_analysis")
    os.makedirs(outdir, exist_ok=True)

    # We'll accumulate per‐layer results here
    results = {}
    cluster_matrix = np.zeros((unique_idx.size, n_layers), dtype=int)

    # For each layer, stream in only that layer's activations
    for layer in tqdm(range(n_layers), desc="Analyzing Layers"):
        # 1) Concatenate this layer across all shards
        acts_list = []
        for f in shard_files:
            arr = np.load(f)['activations']         # shape (shard_size, n_layers, d_model)
            acts_list.append(arr[:, layer, :])
        acts = np.concatenate(acts_list, axis=0)    # shape (total, d_model)

        # 2) Compute avg‐group cosine
        inf_labels = pd.Categorical(df["Inflection Label"]).codes
        lex_labels = pd.Categorical(df["Lemma"]).codes
        def avg_group_cosine(acts, labels):
            total, count = 0.0, 0
            normed = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
            for lab in np.unique(labels):
                idxs = np.where(labels == lab)[0]
                if idxs.size < 2: continue
                sim = normed[idxs] @ normed[idxs].T
                total += sim.sum() - np.trace(sim)
                count += idxs.size * (idxs.size - 1)
            return total/count if count>0 else 0.0

        ci = avg_group_cosine(acts, inf_labels)
        cl = avg_group_cosine(acts, lex_labels)
        results[layer] = {"cosine_inflection": ci, "cosine_lexeme": cl}

        # 3) t-SNE + clustering on unique sentences only
        acts_u = acts[unique_idx]
        pca_proj = PCA(n_components=len(np.unique(inf_labels)), random_state=42).fit_transform(acts_u)
        tsne_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca_proj)

        clusters = KMeans(
            n_clusters=len(np.unique(inf_labels)),
            random_state=config.CLUSTERING["random_state"]
        ).fit_predict(tsne_proj)
        cluster_matrix[:, layer] = clusters

        # 4) Plot and save
        cmap = plt.get_cmap("tab10")
        colors = [cmap(inf_labels[i] % 10) for i in unique_idx]
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, s=10)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=l,
                       markerfacecolor=cmap(i % 10), markersize=6)
            for i, l in enumerate(pd.Categorical(df["Inflection Label"]).categories)
        ]
        plt.legend(handles=handles, title="Inflection", fontsize='small', markerscale=0.7)
        plt.title(f"Layer {layer} t-SNE")
        plt.savefig(os.path.join(outdir, f"layer_{layer}_tsne.png"), bbox_inches="tight")
        plt.close()

    # Save cluster assignments
    rows = []
    for i, idx in enumerate(unique_idx):
        row = {
            "Index": int(idx),
            "Sentence": sentences[idx],
            "Inflection_Label": df.loc[idx, "Inflection Label"]
        }
        for layer in range(n_layers):
            row[f"Cluster_L{layer}"] = int(cluster_matrix[i, layer])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "clusters_per_layer.csv"), index=False)
    np.savez_compressed(os.path.join(outdir, "analysis_results.npz"), results=results)

    # Plot cosine curves
    layers = sorted(results.keys())
    cos_inf = [results[l]["cosine_inflection"] for l in layers]
    cos_lex = [results[l]["cosine_lexeme"] for l in layers]
    plt.figure(figsize=(6, 4))
    plt.plot(layers, cos_inf, marker="o", label="Inflection")
    plt.plot(layers, cos_lex, marker="s", label="Lexeme")
    plt.xlabel("Layer"); plt.ylabel("Avg Cosine Similarity")
    plt.legend()
    plt.savefig(os.path.join(outdir, "cosine_similarity_over_layers.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", required=True,
                   help="Directory of activation shard npz files.")
    p.add_argument("--labels", required=True)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    unsupervised_layer_analysis(args.activations_dir, args.labels, args.model, args.dataset)