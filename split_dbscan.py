#!/usr/bin/env python
"""Perform sklearn DBSCAN clustering in chunks and merge the results
author: tadasnik
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder


def normalize_labels(labels: np.ndarray, labels_increment: int) -> np.ndarray:
    """Normalize labels and increment by labels_increment but preserve -1s."""
    if labels.min() < -1:
        raise ValueError("Labels can not contain values < -1")
    label_encoder = LabelEncoder()
    labels_norm = label_encoder.fit_transform(labels)
    if labels.min() == -1:
        labels_norm += labels_increment
    else:
        labels_norm += labels_increment + 1
    # set -1 back
    labels_norm[labels == -1] = -1
    return labels_norm.astype(int)


class SplitDBSCAN(DBSCAN):
    """Extends sklearn DBSCAN to enable chunked clustering

    Parameters
    ----------

    DBSCAN parameters, plus:

    edge_eps : float, default=0.5
        The threashold distance between any sample (point) of the cluster
        from the chunk edge along the split_dim for that cluster to be
        considered as potentially active or inactive (completed).
        Clusters with any of the samples (points) closer than edge_eps
        to the chunk edge will be considered active
        and added for repeated clustering to the next chunk.
        In most cases the parameter should be equal to eps, but in
        some cases edge_eps may be larger than eps.

    split_dim : int, default=0
        Index of the dimension along which chunking will be performed.
        Note that this refers to a point cloud dimension (index of column
        or feature of the input X array) not a dimension of the X itself.
    """

    def __init__(
        self,
        eps=0.5,
        edge_eps=0.5,
        split_dim=0,
        min_samples=5,
        metric="euclidean",
    ):
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        self.edge_eps = edge_eps
        self.split_dim = split_dim

    def chunk_fit(self, X: np.ndarray, chunk_size: int):
        """A method for performing DBSCAN clustering in chunks

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        chunk_size: int
            Clustering will be performed on X split into chunks of this size.
        """
        # check if X is sorted along the split_dim
        if not np.all(X[:, self.split_dim][:-1] <= X[:, self.split_dim][1:]):
            raise ValueError(
                "The dataset must be monotonically increasing (sorted) along the split_dim."
            )

        # Create an index array of shape X sample_n
        X_index = np.arange(X.shape[0])
        # compute number of chunks based on chunk_size
        chunk_n = int(np.ceil(X_index.shape[0] / chunk_size))
        # split the index array into chunks
        chunks = np.array_split(X_index, chunk_n, axis=0)
        active_index = None
        labels_complete = np.full(X_index.shape[0], -2)
        labels_increment = 0
        for nr, chunk_index in enumerate(chunks, 1):
            if active_index is not None:
                chunk_index = np.hstack([active_index, chunk_index])
            self.fit(X[chunk_index])
            active_mask = self.active_mask(X[chunk_index])
            labels_norm = normalize_labels(self.labels_, labels_increment)
            if nr == len(chunks):
                # If last chunk consider all clusters complete
                complete_index = chunk_index
            else:
                complete_index = chunk_index[~active_mask]
            labels_complete[complete_index] = labels_norm[complete_index]
            labels_increment = labels_complete.max()
            if True in active_mask:
                active_index = chunk_index[active_mask]
                if False not in active_mask:
                    print(
                        """All clusters are "active", attemting to cluster
                        with the next chunk"""
                    )
            else:
                active_index = None
        self.labels_ = labels_complete.astype(int)
        self.active_ = active_index

    def active_mask(self, chunk: np.ndarray) -> np.ndarray[bool]:
        """Splits clusters into completed and active parts.
        The group membership of active points may change when
        clustering the following chunk or with influx of new data.

        Parameters
        ----------
        chunk : {array-like} of shape (n_samples, n_features),
            Training instances used when invocing the fit method.

        Returns:
            active_mask : (bool) a mask with True values indicating
            self.labels_ of active clusters.
        """
        # chunk_edge represents max value in chunk along the
        # split_dimension
        chunk_edge = chunk[:, self.split_dim].max()
        # whithin reach is a mask of samples which are within edge_eps
        # distance from the chunk edge
        within_reach = chunk[:, self.split_dim] >= (chunk_edge - self.edge_eps)
        # unique labels of all within reach samples
        active_labels = np.unique(self.labels_[within_reach])
        # mask indicating all within reach self.labels_
        active_mask = np.isin(self.labels_, active_labels)
        return active_mask
