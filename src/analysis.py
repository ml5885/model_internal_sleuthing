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

def unsupervised_layer_analysis(activations_file, labels_file, model_key, dataset_key):
    data = np.load(activations_file)["activations"]
    _, n_layers, _ = data.shape

    df = pd.read_csv(labels_file)
    cat_inf = pd.Categorical(df["Inflection Label"])
    inf_labels = cat_inf.codes
    inf_names = list(cat_inf.categories)

    cat_lex = pd.Categorical(df["Lemma"])
    lex_labels = cat_lex.codes

    sentences = df["Sentence"].values
    _, unique_idx = np.unique(sentences, return_index=True)
    unique_idx = np.sort(unique_idx)
    inf_u = inf_labels[unique_idx]

    results = {}
    cluster_matrix = np.zeros((unique_idx.size, n_layers), dtype=int)

    outdir = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset_key}_analysis")
    os.makedirs(outdir, exist_ok=True)

    for layer in tqdm(range(n_layers), desc="Analyzing Layers"):
        acts = data[:, layer, :]

        ci = avg_group_cosine(acts, inf_labels)
        cl = avg_group_cosine(acts, lex_labels)
        results[layer] = {"cosine_inflection": ci, "cosine_lexeme": cl}

        acts_u = acts[unique_idx]
        pca_proj = PCA(n_components=len(inf_names), random_state=42).fit_transform(acts_u)
        tsne_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca_proj)

        clusters = KMeans(n_clusters=len(inf_names), random_state=config.CLUSTERING["random_state"])\
            .fit_predict(tsne_proj)
        cluster_matrix[:, layer] = clusters

        cmap = plt.get_cmap("tab10")
        colors = [cmap(inf_u[i] % 10) for i in range(len(inf_u))]

        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, s=10)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=inf_names[i],
                       markerfacecolor=cmap(i % 10), markersize=6)
            for i in range(len(inf_names))
        ]
        plt.legend(handles=handles, title="Inflection", fontsize='small', markerscale=0.7)
        plt.title(f"Layer {layer} t-SNE")
        plt.savefig(os.path.join(outdir, f"layer_{layer}_tsne.png"), bbox_inches="tight")
        plt.close()

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

    layers = list(results.keys())
    cos_inf = [results[l]["cosine_inflection"] for l in layers]
    cos_lex = [results[l]["cosine_lexeme"] for l in layers]

    plt.figure(figsize=(6, 4))
    plt.plot(layers, cos_inf, marker="o", label="Inflection")
    plt.plot(layers, cos_lex, marker="s", label="Lexeme")
    plt.xlabel("Layer")
    plt.ylabel("Avg Cosine Similarity")
    plt.legend()
    plt.savefig(os.path.join(outdir, "cosine_similarity_over_layers.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    unsupervised_layer_analysis(args.activations, args.labels, args.model, args.dataset)
