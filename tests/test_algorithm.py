import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from src.dbscan_project.algorithm import CustomDBSCAN

def test_basic_clustering():
    # generate 50 balls, 2 groups
    # random state to get same result every time
    X, _ = make_blobs(n_samples=50, centers=2, random_state=42, cluster_std=0.5)

    # large radius (1.0) to catch points easily, min 5 neighbors
    model = CustomDBSCAN(eps=1.0, min_samples=5)
    model.fit(X)

    # set removes duplicates so i see how many groups i got, must be > 1
    assert len(set(model.labels_)) > 1
    
    # checking if every point got a label (length check)
    assert len(model.labels_) == 50


# testing moons
def test_moons_structure():
    # generating moon shapes
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

    # parameters picked by trial and error for this shape
    model = CustomDBSCAN(eps=0.3, min_samples=5)
    model.fit(X)
    
    # getting unique labels
    unique_labels = set(model.labels_)
    
    # if it found noise (-1) remove it from count
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # must remain exactly 2 groups (because 2 moons)
    assert len(unique_labels) == 2


# checking noise, points that are far away
def test_noise_handling():
    # manual coords: 3 close together and one far away (100, 100)
    X = np.array([[0, 0], [0, 0.1], [0.1, 0], [100, 100]])

    # min_samples 3 so these close ones make a cluster
    model = CustomDBSCAN(eps=0.5, min_samples=3)
    model.fit(X)

    # this last point (far) must be -1 so noise
    assert model.labels_[-1] == -1
    
    # first point is in group so cant be noise
    assert model.labels_[0] != -1


# weird cases so the app doesnt crash
def test_edge_cases():
    # empty data list
    X_empty = np.zeros((0, 2))
    model = CustomDBSCAN(eps=0.5, min_samples=5)
    model.fit(X_empty)
    
    # if no data then label list should be empty
    assert len(model.labels_) == 0

    # just one point
    X_single = np.array([[1, 1]])
    model.fit(X_single)
    
    # one point is not enough for cluster so must be noise (-1)
    assert model.labels_[0] == -1


# comparing with sklearn to see if results are the same
def test_compare_with_sklearn():
    # 100 points, 3 groups
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    # saving params in dict so they are identical in both
    params = {'eps': 0.5, 'min_samples': 5}

    # my algo
    custom = CustomDBSCAN(**params)
    custom.fit(X)

    # sklearn algo (reference)
    sklearn = SklearnDBSCAN(**params)
    sklearn_labels = sklearn.fit_predict(X)

    # calculating ARI, checks if clusters are identical (1.0 is perfect)
    ari = adjusted_rand_score(custom.labels_, sklearn_labels)

    # must be identical
    assert ari == 1.0