#!/usr/bin/env python
"""Perform sklearn self clustering in chunks and merge the results
author: tadasnik
"""
import numpy as np

from numpy.lib import recfunctions
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder


def normalize_labels(labels: np.ndarray, labels_increment) -> np.ndarray:
    """Normalize labels and increment by labels_increment but preserve -1s."""
    if labels.min() < -1:
        raise ValueError("Labels can not contain values < -1")
    labels_norm = labels.copy()
    mask = labels != -1
    label_encoder = LabelEncoder()
    labels_transformed = label_encoder.fit_transform(labels[mask])
    #mask_1s = labels != -1
    assert labels_transformed is not None
    labels_transformed += labels_increment
    labels_norm[mask] = labels_transformed
    return labels_norm.astype(int)


class SplitDBSCAN(cluster.DBSCAN):
    """Extends sklearn self to enable chunked clustering

    Parameters
    ----------

    self parameters, plus:

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

    chunk_size: int
            Clustering will be performed on X split into chunks of this size.
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        edge_eps=0.5,
        split_dim=0,
        chunk_size=1e6,
        metric="euclidean",
    ):
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        self.edge_eps = edge_eps
        self.split_dim = split_dim
        self.chunk_size = chunk_size

    def fit(self, X: np.ndarray):
        """A method for performing self clustering in chunks

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.
        """

        # Create a structured array to store X index and cluster labels
        inds = np.zeros((X.shape[0],), [("index", int), ("labels", int)])

        if not np.all(X[:, self.split_dim][:-1] <= X[:, self.split_dim][1:]):
            print("not sorted")
            inds["index"] = X[:, self.split_dim].argsort()
        else:
            inds["index"] = np.arange(X.shape[0])
        inds["labels"] = -1
        # compute number of chunks based on chunk_size
        chunk_n = int(np.ceil(len(inds) / self.chunk_size))
        # split the index array into chunks
        chunks = np.array_split(np.arange(len(inds["index"])), chunk_n, axis=0)
        chunk_inds = [(x[0], x[-1]) for x in chunks]

        for chunk_number, (start, end) in enumerate(chunk_inds, 1):
            eps_start = start
            if chunk_number > 1:
                chunk_edge = X[inds["index"][start]][self.split_dim]
                # Determine overlap points scale (extend) eps overlap by min_samples
                eps_edge = chunk_edge - self.eps * self.min_samples
                eps_start = np.searchsorted(
                    X[:, 0], eps_edge, sorter=inds["index"], side="right"
                )
            chunk_index = inds["index"][eps_start : end + 1]
            # chunk = X[inds['index'][start:end]]
            self.fit(X[chunk_index])
            if (chunk_number > 1) and (eps_start != start):
                eps_old = inds[eps_start:start]
                temp = np.zeros((eps_old.shape[0],), [("index", int), ("labels", int)])
                temp["index"] = chunk_index[: eps_old.shape[0]]
                temp["labels"] = self.labels_[: eps_old.shape[0]]
                # eps_new = temp[eps_old['index']]
                merged = recfunctions.join_by(
                    "index", temp, eps_old, "inner", usemask=False
                )
                if len(merged) > 0:
                    # step 1: Drop -1 in relabeled dataset (so that mapping from
                    # earlier cluster member to no cluster membership in current
                    # chunk is not performed for any of the points in the overlap).
                    merged = merged[merged["labels2"] != -1]
                    # step 1: Remap points in the overlap that have changed from -1
                    # to cluster membership.
                    selected = merged[merged["labels1"] == -1]

                    mask_to_1s = np.isin(
                        inds["index"][eps_start : end + 1], selected["index"]
                    )
                    merged = merged[merged["labels1"] != -1]
                    if len(merged) > 0:
                        mapping = np.unique(merged[["labels1", "labels2"]])
                        keys = np.array([x[0] for x in mapping])
                        values = np.array([x[1] for x in mapping])
                        # check for repeated keys and if present remove from keys/values
                        # and relabel afected clusters in the previous chunks.
                        # len(set(keys)) check is quicker for small arrays
                        # but np.unique would be still needed if True, so skipping for now.
                        # if len(set(keys)) != len(keys):
                        _, ind_start, count = np.unique(
                            keys, return_counts=True, return_index=True
                        )
                        if np.any(count > 1):
                            # relabel newly merged clusters. Currenty nested python loops,
                            # perhaps could be vectorised if slow for large number of clusters.
                            for repeat_ind in ind_start[count > 1]:
                                r_values = values[
                                    repeat_ind : repeat_ind + count[repeat_ind]
                                ]
                                for rep_value in r_values[1:]:
                                    inds["labels"][
                                        inds["labels"] == rep_value
                                    ] = r_values[0]
                                    values[values == rep_value] = r_values[0]
                            # keep only unique
                            keys = keys[ind_start]
                            values = values[ind_start]
                        # sort_idx = np.argsort(keys)
                        idx = np.searchsorted(keys, self.labels_)

                        idx[idx == len(values)] = 0
                        mask = keys[idx] == self.labels_
                        labels = np.where(mask, values[idx], self.labels_)
                        if not np.all(mask):
                            labels[~mask] = normalize_labels(
                                labels[~mask], inds["labels"].max()
                            )
                        # check for repeated keys and relabel merged clustersn
                    else:
                        labels = normalize_labels(self.labels_, inds["labels"].max())
                    inds["labels"][eps_start : end + 1][~mask_to_1s] = labels[
                        ~mask_to_1s
                    ]
                else:
                    inds["labels"][start : end + 1] = self.labels_
            else:
                inds["labels"][start : end + 1] = self.labels_

        self.labels_ = inds["labels"].astype(int)

    def active_mask(self, chunk: np.ndarray) -> np.ndarray[bool]:
        """Splits clusters into completed and active parts.
        The group membership of active points may change when
        clustering the following chunk.

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
        # whithin reach is a mask of samples which are < edge_eps
        # distance from the chunk edge
        within_reach = chunk[:, self.split_dim] >= (chunk_edge - self.edge_eps)
        # unique labels of all within reach samples
        active_labels = np.unique(self.labels_[within_reach])
        # mask indicating all within reach self.labels_
        active_mask = np.isin(self.labels_, active_labels)
        return active_mask
