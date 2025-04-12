import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from src import config, utils

def compute_cosine_sims(activations, labels):
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

def unsupervised_layer_analysis(activations_file, labels_file, model_key, dataset_key):
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    utils.log_info(f"Loaded activations shape: {data.shape}")

    df = pd.read_csv(labels_file)
    # Prepare inflection labels as strings
    inflection_labels_raw = df["Inflection Label"].values
    unique_inflections = sorted(set(inflection_labels_raw))
    # Map each inflection label to an integer
    inflection_to_idx = {inf: idx for idx, inf in enumerate(unique_inflections)}
    inflection_labels = np.array([inflection_to_idx[inf] for inf in inflection_labels_raw])
    
    # Prepare lexeme labels (but for PCA we can still use them as numerical codes)
    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])
    
    results = {}
    # Updated analysis folder: now includes the model key.
    analysis_folder = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset_key}_analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    
    # For discrete colormap: choose a colormap with as many colors as inflection classes.
    # cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(unique_inflections)])
    
    for layer in tqdm(range(n_layers), desc="Analyzing Layers"):
        acts = data[:, layer, :]
        cosine_inflection = compute_cosine_sims(acts, inflection_labels)
        cosine_lexeme = compute_cosine_sims(acts, lexeme_labels)
        # clusters = run_clustering(acts, n_clusters=config.CLUSTERING["n_clusters"])
        # fake clusters
        clusters = np.random.randint(0, len(unique_inflections), size=acts.shape[0])
        results[layer] = {
            "cosine_inflection": cosine_inflection,
            "cosine_lexeme": cosine_lexeme,
            "cluster_labels": clusters
        }
        utils.log_info(f"Layer {layer}: Cosine (inflection) = {cosine_inflection:.4f}, Cosine (lexeme) = {cosine_lexeme:.4f}")

        # # Generate PCA plot for each layer with discrete inflection labels.
        # try:
        #     pca = PCA(n_components=2)
        #     projected = pca.fit_transform(acts)
        #     plt.figure(figsize=(8,6))
            
        #     for idx, inflection in enumerate(unique_inflections):
        #         mask = inflection_labels == idx
        #         plt.scatter(projected[mask, 0], projected[mask, 1], 
        #                   c=[plt.cm.tab10(idx/10)],
        #                   label=inflection,
        #                   s=20)
            
        #     plt.title(f"Layer {layer} PCA (Inflection)")
        #     plt.xlabel("PCA Component 1")
        #     plt.ylabel("PCA Component 2")
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        #     plot_path = os.path.join(analysis_folder, f"layer_{layer}_pca_inflection.png")
        #     plt.savefig(plot_path, bbox_inches='tight')  # bbox_inches='tight' to prevent legend cutoff
        #     plt.close()
        #     utils.log_info(f"Saved PCA plot for layer {layer} to {plot_path}")
        # except Exception as e:
        #     utils.log_error(f"PCA plotting failed for layer {layer}: {str(e)}")
    
    output_path = os.path.join(analysis_folder, "analysis_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Unsupervised analysis results saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Perform unsupervised analysis on activations.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the CSV file with labels.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model key (used for saving results).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset label (e.g., controlled, wikitext, combined).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    unsupervised_layer_analysis(args.activations, args.labels, args.model, args.dataset)
