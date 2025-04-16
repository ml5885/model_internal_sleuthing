import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm
from src import config, utils
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_sims(activations, labels):
    # Compute cosine similarity over datapoints with matching labels.
    sims = cosine_similarity(activations)
    n = activations.shape[0]
    sim_total = 0.0
    count = 0
    progress_interval = max(1, n // 10)
    for i in range(n):
        if i % progress_interval == 0:
            print(f"  Row {i}/{n} processed")
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                sim_total += sims[i, j]
                count += 1
    avg_sim = sim_total / count if count > 0 else 0.0
    return avg_sim

def unsupervised_layer_analysis(activations_file, labels_file, model_key, dataset_key):
    print("Loading activations...")
    utils.log_info(f"Loading activations from {activations_file}")
    data = np.load(activations_file)["activations"]
    n_examples, n_layers, d_model = data.shape
    print(f"Activations shape: {data.shape}")
    utils.log_info(f"Activations shape: {data.shape}")

    df = pd.read_csv(labels_file)
    utils.log_info(f"Labels loaded from {labels_file}")
    
    # Prepare labels.
    inflection_labels_raw = df["Inflection Label"].values
    unique_inflections = sorted(set(inflection_labels_raw))
    inflection_to_idx = {inf: idx for idx, inf in enumerate(unique_inflections)}
    inflection_labels = np.array([inflection_to_idx[inf] for inf in inflection_labels_raw])
    
    lexemes = df["Lemma"].values
    lexeme_to_idx = {lex: idx for idx, lex in enumerate(sorted(set(lexemes)))}
    lexeme_labels = np.array([lexeme_to_idx[lex] for lex in lexemes])
    
    utils.log_info(f"Unique inflections: {unique_inflections} ({len(unique_inflections)} classes)")
    utils.log_info(f"Unique lexemes: {list(lexeme_to_idx.keys())} ({len(lexeme_to_idx)} classes)")
    
    results = {}
    analysis_folder = os.path.join(config.OUTPUT_DIR, f"{model_key}_{dataset_key}_analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    print(f"Saving analysis results to: {analysis_folder}")

    # Process each layer.
    for layer in tqdm(range(n_layers), desc="Analyzing Layers"):
        # Minimal per-layer print summary.
        acts = data[:, layer, :]
        cosine_inflection = compute_cosine_sims(acts, inflection_labels)
        cosine_lexeme = compute_cosine_sims(acts, lexeme_labels)
        results[layer] = {
            "cosine_inflection": cosine_inflection,
            "cosine_lexeme": cosine_lexeme
        }
        utils.log_info(f"Layer {layer}: CosineInflection={cosine_inflection:.4f}, CosineLexeme={cosine_lexeme:.4f}")
        
        # Run t-SNE.
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_proj = tsne.fit_transform(acts)
        
        # KMeans clustering.
        n_clusters = len(unique_inflections)
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.CLUSTERING["random_state"])
        cluster_labels = kmeans.fit_predict(tsne_proj)
        
        # Save CSV with minimal information.
        csv_rows = []
        for idx in range(n_examples):
            csv_rows.append({
                "Index": idx,
                "Lemma": df.loc[idx, "Lemma"] if "Lemma" in df.columns else "",
                "Inflection_Label": df.loc[idx, "Inflection Label"] if "Inflection Label" in df.columns else "",
                "TSNE1": tsne_proj[idx, 0],
                "TSNE2": tsne_proj[idx, 1],
                "Cluster": int(cluster_labels[idx])
            })
        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(analysis_folder, f"layer_{layer}_clusters.csv")
        csv_df.to_csv(csv_path, index=False)
        
        # Plot t-SNE scatter (without centroids).
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=cluster_labels, cmap="viridis", s=20)
        plt.title(f"Layer {layer} t-SNE (Inflection Clusters)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.colorbar(scatter, label="Cluster")
        tsne_plot_path = os.path.join(analysis_folder, f"layer_{layer}_tsne_inflection.png")
        plt.savefig(tsne_plot_path, bbox_inches="tight")
        plt.close()
        
        print(f"Layer {layer} - Avg Cosine Inflection: {cosine_inflection:.4f}, saved CSV and t-SNE plot.")
    
    # Save analysis results.
    output_path = os.path.join(analysis_folder, "analysis_results.npz")
    np.savez_compressed(output_path, results=results)
    utils.log_info(f"Analysis results saved to {output_path}")

    # Combined cosine similarity plot.
    layers = list(range(n_layers))
    cosine_inflections = [results[layer]["cosine_inflection"] for layer in layers]
    cosine_lexemes = [results[layer]["cosine_lexeme"] for layer in layers]
    plt.figure(figsize=(8, 6))
    plt.plot(layers, cosine_inflections, marker="o", label="Cosine Inflection")
    plt.plot(layers, cosine_lexemes, marker="s", label="Cosine Lexeme")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity over Layers")
    plt.legend()
    combined_plot_path = os.path.join(analysis_folder, "cosine_similarity_over_layers.png")
    plt.savefig(combined_plot_path, bbox_inches="tight")
    plt.close()
    print("Analysis complete. Combined cosine similarity plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform unsupervised analysis on activations using t-SNE.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the NPZ file with activations.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the CSV file with labels.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model key (used for saving results).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset label (e.g., controlled, wikitext, combined).")
    args = parser.parse_args()
    
    print("Starting analysis:")
    print(f" Activations: {args.activations}")
    print(f" Labels: {args.labels}")
    print(f" Model: {args.model}")
    print(f" Dataset: {args.dataset}")
    
    unsupervised_layer_analysis(args.activations, args.labels, args.model, args.dataset)
    print("Unsupervised analysis complete.")
