import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from dbscan_project.algorithm import CustomDBSCAN


def test_basic_clustering():
    """Checks if the algorithm identifies obvious clusters correctly."""
    # Generate simple blob data
    X, _ = make_blobs(n_samples=50, centers=2, random_state=42, cluster_std=0.5)

    model = CustomDBSCAN(eps=1.0, min_samples=5)
    model.fit(X)

    # Assert that we found valid clusters (labels other than -1)
    assert len(set(model.labels_)) > 1
    # Assert output shape matches input
    assert len(model.labels_) == 50


def test_noise_handling():
    """Checks if distant outliers are correctly classified as noise (-1)."""
    # 3 points close to each other, 1 point very far away
    X = np.array([[0, 0], [0, 0.1], [0.1, 0], [100, 100]])

    # Min samples = 3, so the cluster of 3 should form, but the outlier should not
    model = CustomDBSCAN(eps=0.5, min_samples=3)
    model.fit(X)

    # The last point (index 3) must be noise (-1)
    assert model.labels_[-1] == -1
    # The first point must belong to a cluster (e.g., 0)
    assert model.labels_[0] != -1


def test_compare_with_sklearn():
    """
    Compares the results of the custom implementation against Scikit-Learn.
    They should be mathematically identical for small datasets.
    """
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    params = {'eps': 0.5, 'min_samples': 5}

    # Run Custom Implementation
    custom = CustomDBSCAN(**params)
    custom.fit(X)

    # Run Scikit-Learn Implementation
    sklearn = SklearnDBSCAN(**params)
    sklearn_labels = sklearn.fit_predict(X)

    # Calculate Adjusted Rand Index (ARI)
    # ARI = 1.0 means the clusterings are identical (ignoring permutation of labels)
    ari = adjusted_rand_score(custom.labels_, sklearn_labels)

    assert ari == 1.0