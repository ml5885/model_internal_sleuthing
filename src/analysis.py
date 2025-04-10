import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src import config, utils

def compute_cosine_similarities(activations, labels):
    sim_total = 0.0
    count = 0
    sims = cosine_similarity(activations)
    n = activations.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                sim_total += sims[i, j]
                count += 1
    return sim_total / count if count > 0 else 0.0

def run_clustering(activations, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.CLUSTERING["random_state"])
    cluster_labels = kmeans.fit_predict(activations)
    return cluster_labels

def unsupervised_layer_analysis(activations_file, labels_file):
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations shape: {data.shape}")

    df = pd.read_csv(labels_file)
    tense_labels = df["Inflection Label"].map(lambda x: 1 if x.lower() == "past" else 0).values
    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])

    results = {}
    for layer in range(n_layers):
        acts = data[:, layer, :]
        cosine_tense = compute_cosine_similarities(acts, tense_labels)
        cosine_lexeme = compute_cosine_similarities(acts, lexeme_labels)
        clusters = run_clustering(acts, n_clusters=config.CLUSTERING["n_clusters"])
        results[layer] = {
            "cosine_tense": cosine_tense,
            "cosine_lexeme": cosine_lexeme,
            "cluster_labels": clusters
        }
        utils.log_info(f"Layer {layer}: Cosine similarity (tense) = {cosine_tense:.4f}, (lexeme) = {cosine_lexeme:.4f}")

        # Generate PCA plots (colored by tense)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(acts)
            plt.figure()
            plt.scatter(projected[:, 0], projected[:, 1], c=tense_labels, cmap="viridis", s=10)
            plt.title(f"Layer {layer} PCA (colored by tense)")
            plot_path = os.path.join(config.OUTPUT_DIR, f"layer_{layer}_pca_tense.png")
            plt.savefig(plot_path)
            plt.close()
            utils.log_info(f"Saved PCA plot for layer {layer} to {plot_path}")
        except Exception as e:
            utils.log_error(f"PCA plotting failed for layer {layer}: {str(e)}")

    output_path = os.path.join(config.OUTPUT_DIR, "unsupervised_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Unsupervised analysis results saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Perform unsupervised analysis on activations.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="CSV file with labels.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    unsupervised_layer_analysis(args.activations, args.labels)
