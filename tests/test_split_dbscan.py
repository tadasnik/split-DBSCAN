from re import A
import pytest
import numpy as np
from split_dbscan import normalize_labels, SplitDBSCAN


def test_normalize_labels():
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
