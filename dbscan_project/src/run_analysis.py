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
        # Supervised Metrics (requiring ground truth)
        "ARI (Supervised)": round(adjusted_rand_score(y_true, labels), 4),
        "NMI (Supervised)": round(normalized_mutual_info_score(y_true, labels), 4),
        # Unsupervised Metrics (internal geometry)
        "Silhouette (Unsupervised)": round(silhouette_score(X, labels), 4),
        "Davies-Bouldin (Unsupervised)": round(davies_bouldin_score(X, labels), 4)
    }


def run_experiment():
    print("--- Starting DBSCAN Benchmark Project ---\n")

    # 1. Generate Dataset (Moons - ideal for density-based clustering)
    print("Generating 'Make Moons' dataset (N=2000)...")
    X, y_true = make_moons(n_samples=2000, noise=0.1, random_state=42)

    # Hyperparameters
    PARAMS = {'eps': 0.15, 'min_samples': 10}

    results = []

    # 2. Run Custom Implementation
    print("Running Custom DBSCAN implementation...")
    start_time = time.time()
    custom_model = CustomDBSCAN(**PARAMS)
    custom_model.fit(X)
    custom_time = time.time() - start_time

    results.append(evaluate_model(
        "Custom Implementation", custom_model.labels_, X, y_true, custom_time
    ))

    # 3. Run Scikit-Learn Implementation
    print("Running Scikit-Learn implementation...")
    start_time = time.time()
    sklearn_model = DBSCAN(**PARAMS)
    sk_labels = sklearn_model.fit_predict(X)
    sk_time = time.time() - start_time

    results.append(evaluate_model(
        "Scikit-Learn (Built-in)", sk_labels, X, y_true, sk_time
    ))

    # 4. Display Results
    df_results = pd.DataFrame(results)
    print("\n=== PERFORMANCE COMPARISON ===")
    print(df_results.to_string(index=False))

    # 5. Simple Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Custom Plot
    ax1.scatter(X[:, 0], X[:, 1], c=custom_model.labels_, cmap='Spectral', s=5)
    ax1.set_title(f"Custom DBSCAN (Time: {custom_time:.4f}s)")

    # Sklearn Plot
    ax2.scatter(X[:, 0], X[:, 1], c=sk_labels, cmap='Spectral', s=5)
    ax2.set_title(f"Scikit-Learn (Time: {sk_time:.4f}s)")

    plt.show()


if __name__ == "__main__":
    run_experiment()