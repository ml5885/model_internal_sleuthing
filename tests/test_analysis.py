import numpy as np
from src import analysis

def test_cosine_similarity():
    print("Starting test: test_cosine_similarity")
    # Generate random matrix
    X = np.random.randn(10, 768)
    labels = np.random.randint(0, 2, size=(10,))
    print("Generated random data for cosine similarity computation.")

    # Compute average cosine similarity for data points with same labels
    avg_sim = analysis.compute_cosine_similarities(X, labels)
    print(f"Computed average cosine similarity: {avg_sim}")

    # Check for valid cosine similarity value
    assert -1.0 <= avg_sim <= 1.0, f"Average similarity {avg_sim} is out of expected range [-1, 1]"
    print("test_cosine_similarity passed.\n")

def test_kmeans_clustering():
    print("Starting test: test_kmeans_clustering")
    # Generate random feature matrix
    X = np.random.randn(20, 768)
    print("Generated random data for clustering.")

    # Run clustering algorithm
    clusters = analysis.run_clustering(X, n_clusters=2)
    print(f"Clustering completed. Cluster labels shape: {clusters.shape}")

    # Expecting one cluster assignment per sample
    assert clusters.shape[0] == 20, f"Expected 20 cluster assignments, got {clusters.shape[0]}"
    print("test_kmeans_clustering passed.\n")
