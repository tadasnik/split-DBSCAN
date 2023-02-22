import pytest

import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from split_dbscan import normalize_labels, SplitDBSCAN


def toy_datasets():
    """Return example datasets with configuration parameters"""
    n_samples = 500
    random_state = 170
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    test_datasets = [
        (
            noisy_circles,
            {
                "min_samples": 7,
            },
        ),
        (
            noisy_moons,
            {
                "min_samples": 7,
            },
        ),
        (
            varied,
            {
                "eps": 0.18,
                "min_samples": 7,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "min_samples": 7,
            },
        ),
        (blobs, {"min_samples": 7}),
        (no_structure, {}),
    ]
    return test_datasets


def test_normalize_labels():
    """Test label normalization function"""
    labels_increment = 3
    labels_test = np.array([0, 1, 2, 2, 3, 4, -1, 4, -1, 0, -1])
    labels = np.array([2, 8, 1, 1, 7, 9, -1, 9, -1, 2, -1])
    labels_norm = normalize_labels(labels, labels_increment)
    # The number of unique labels
    assert len(np.unique(labels_norm)) == len(np.unique(labels_test))
    # The number of -1s
    assert len(labels_norm[labels_norm == -1]) == len(labels_test[labels_test == -1])
    # Minimum label is equal to labels_increment
    assert labels_norm[labels_norm != -1].min() == labels_increment


def test_fit_toy_datasets():
    """Wall to wall test against vanilla DBSCAN with scikit-learn example datasets"""
    chunk_size = 150

    default_base = {
        "eps": 0.3,
        "min_samples": 7,
    }
    test_datasets = toy_datasets()
    for nr, (dataset, algo_params) in enumerate(test_datasets):
        params = default_base.copy()
        params.update(algo_params)
        X, _ = dataset
        X = StandardScaler().fit_transform(X)
        dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        split_dbscan = SplitDBSCAN(
            eps=params["eps"], min_samples=params["min_samples"], chunk_size=chunk_size
        )
        dbscan.fit(X)
        split_dbscan.fit(X)
        assert len(np.unique(dbscan.labels_)) == len(np.unique(split_dbscan.labels_))
        assert len(dbscan.labels_[dbscan.labels_ == -1]) == len(
            split_dbscan.labels_[split_dbscan.labels_ == -1]
        )
