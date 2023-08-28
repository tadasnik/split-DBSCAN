import pytest

import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from split_dbscan import normalize_labels, SplitDBSCAN

from tests import test_datasets


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
    datasets = test_datasets.sklearn_datasets()
    for _, (dataset, algo_params) in enumerate(datasets):
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
