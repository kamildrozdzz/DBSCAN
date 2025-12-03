import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# Import the custom class from the package
from dbscan_project.algorithm import CustomDBSCAN


def evaluate_model(name, labels, X, y_true, exec_time):
    """
    Helper function to calculate metrics and return a dictionary.
    """
    # Check if clustering failed (everything is noise or 1 cluster)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 1:
        return {
            "Algorithm": name,
            "Time (s)": exec_time,
            "Clusters": 0,
            "ARI (Supervised)": "N/A",
            "NMI (Supervised)": "N/A",
            "Silhouette (Unsupervised)": "N/A",
            "Davies-Bouldin (Unsupervised)": "N/A"
        }

    return {
        "Algorithm": name,
        "Time (s)": round(exec_time, 5),
        "Clusters": n_clusters,
        # Supervised Metrics
        "ARI (Supervised)": round(adjusted_rand_score(y_true, labels), 4),
        "NMI (Supervised)": round(normalized_mutual_info_score(y_true, labels), 4),
        # Unsupervised Metrics
        "Silhouette (Unsupervised)": round(silhouette_score(X, labels), 4),
        "Davies-Bouldin (Unsupervised)": round(davies_bouldin_score(X, labels), 4)
    }


def run_experiment():
    print("--- Starting DBSCAN Benchmark Project ---\n")

    # Generate Dataset
    print("Generating 'Make Moons' dataset...")
    X, y_true = make_moons(n_samples=2000, noise=0.1, random_state=42)

    # Hyperparameters
    PARAMS = {'eps': 0.15, 'min_samples': 10}

    results = []

    # Run Custom Implementation
    print("Running Custom DBSCAN implementation...")
    start_time = time.time()
    custom_model = CustomDBSCAN(**PARAMS)
    custom_model.fit(X)
    custom_time = time.time() - start_time

    results.append(evaluate_model(
        "Custom Implementation", custom_model.labels_, X, y_true, custom_time
    ))

    # Run Scikit-Learn Implementation
    print("Running Scikit-Learn implementation...")
    start_time = time.time()
    sklearn_model = DBSCAN(**PARAMS)
    sk_labels = sklearn_model.fit_predict(X)
    sk_time = time.time() - start_time

    results.append(evaluate_model(
        "Scikit-Learn (Built-in)", sk_labels, X, y_true, sk_time
    ))

    # Display Results
    df_results = pd.DataFrame(results)
    print("\n=== PERFORMANCE COMPARISON ===")
    print(df_results.to_string(index=False))

    # Simple Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Custom Plot
    ax1.scatter(X[:, 0], X[:, 1], c=custom_model.labels_, cmap='Spectral', s=5)
    ax1.set_title(f"Custom DBSCAN (Time: {custom_time:.4f}s)")

    # Sklearn Plot
    ax2.scatter(X[:, 0], X[:, 1], c=sk_labels, cmap='Spectral', s=5)
    ax2.set_title(f"Scikit-Learn (Time: {sk_time:.4f}s)")

    plt.show()


def run_scalability_test():
    """
    Tests the performance of algorithms with increasing sample sizes
    and plots a comparative graph of execution time.
    """
    # How many steps
    sample_sizes = [100, 500, 1000, 1500, 2000, 2500, 3000]

    times_custom = []
    times_sklearn = []

    print("\n--- STARTING SCALABILITY TEST (Performance vs Dataset Size) ---")

    # Fixed parameters for the test to ensure consistency
    params = {'eps': 0.2, 'min_samples': 5}

    for n in sample_sizes:
        print(f"Testing for N = {n} samples...")

        # Generate dataset of specific size
        X, _ = make_moons(n_samples=n, noise=0.1, random_state=42)

        # 1. Measure execution time for Custom Implementation
        start = time.time()
        CustomDBSCAN(**params).fit(X)
        times_custom.append(time.time() - start)

        # 2. Measure execution time for Scikit-Learn
        start = time.time()
        DBSCAN(**params).fit(X)
        times_sklearn.append(time.time() - start)

    plt.figure(figsize=(10, 6))

    plt.plot(sample_sizes, times_custom, 'o-', color='red', label='Custom Implementation (Brute-force)')

    plt.plot(sample_sizes, times_sklearn, 's-', color='blue', label='Scikit-Learn (KD-Tree optimized)')

    plt.title("Scalability Analysis: Execution Time vs Dataset Size")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
    run_scalability_test()