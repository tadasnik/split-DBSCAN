#!/usr/bin/env python
"""Perform sklearn DBSCAN clustering in chunks and merge the results
author: tadasnik
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

import numpy.lib.recfunctions as rf

import matplotlib.pyplot as plt

from itertools import cycle, islice


def normalize_labels(labels: np.ndarray, labels_increment) -> np.ndarray:
    """Normalize labels and increment by labels_increment but preserve -1s."""
    if labels.min() < -1:
        raise ValueError("Labels can not contain values < -1")
    label_encoder = LabelEncoder()
    labels_transformed = label_encoder.fit_transform(labels)
    mask_1s = labels != -1
    assert labels_transformed is not None
    labels_transformed[mask_1s] += labels_increment + 1
    labels_transformed[~mask_1s] = -1
    return labels_transformed.astype(int)


class SplitDBSCAN(cluster.DBSCAN):
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
        """A method for performing DBSCAN clustering in chunks

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.
        """
        X_index = np.arange(X.shape[0])
        if not np.all(X[:, self.split_dim][:-1] <= X[:, self.split_dim][1:]):
            print('not sorted')
            X_index_s = X[:, self.split_dim].argsort()
            #raise ValueError(
            #    "The dataset must be monotonically increasing (sorted) along the split_dim."
            #)
        else:
            # Create an index array of shape X sample_n
            X_index_s = X_index
     
       # compute number of chunks based on chunk_size
        chunk_n = int(np.ceil(X_index.shape[0] / self.chunk_size))
        print('chunks no: ', chunk_n)
        # split the index array into chunks
        chunks = np.array_split(X_index, chunk_n, axis=0)
        active_index = None
        labels_complete = np.full(X_index.shape[0], -2)
        labels_increment = 0
        for nr, chunk_index in enumerate(chunks, 1):
            if active_index is not None:
                chunk_index = np.hstack([active_index, chunk_index])
            super().fit(X[chunk_index])
            active_mask = self.active_mask(X[chunk_index])
            labels_norm = normalize_labels(self.labels_, labels_increment)
            if nr == len(chunks):
                complete_index = chunk_index
                labels_complete[complete_index] = labels_norm
            else:
                complete_index = chunk_index[~active_mask]
                labels_complete[complete_index] = labels_norm[~active_mask]
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


if __name__ == '__main__':

    plot_num = 1
    eps = 0.15
    min_samples = 2
    n_samples = 500
# set chunk size
    chunk_n = 5
    chunk_size = int(np.ceil(n_samples / chunk_n))
# Anisotropicly distributed data
    random_state = 170
# blobs with varied variances
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

    default_base = {
        "eps": 0.3,
        "min_samples": 7,
    }

    datasets = [

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
    fig, axs = plt.subplots(len(datasets), chunk_n + 1, figsize=(15, 15))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
    #dataset = noisy_circles
    #algo_params = {
    #        "eps": 0.3,
    #        "min_samples": 7,
    #}
    #i_dataset = 0
   # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset
        #if i_dataset == 5:
        #    X = np.load('no_structure.npy')

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)



        dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"])



        split_dbscan = SplitDBSCAN(eps=eps, min_samples=min_samples, chunk_size=chunk_size)


        # normalize dataset for easier parameter selection

        inds = np.zeros((X.shape[0],),[('index',int),('labels',int)])

        X_index = np.arange(X.shape[0])
        if not np.all(X[:, split_dbscan.split_dim][:-1] <= X[:, split_dbscan.split_dim][1:]):
            print('not sorted')
            inds['index'] = X[:, split_dbscan.split_dim].argsort()
            #raise ValueError(
            #    "The dataset must be monotonically increasing (sorted) along the split_dim."
            #)
        else:
            # Create an index array of shape X sample_n
            inds['index'] = np.arange(X.shape[0])
        # compute number of chunks based on chunk_size
        inds['labels'] = -1
        chunk_n = int(np.ceil(len(inds) / split_dbscan.chunk_size))
        # split the index array into chunks
        chunks = np.array_split(np.arange(len(inds['index'])), chunk_n, axis=0)
        chunk_inds = [(x[0], x[-1]) for x in chunks]

        # active_index = None
        # labels_complete = np.full(X_index.shape[0], -1)
        # labels_increment = 0
        for nr, (start, end) in enumerate(chunk_inds, 1):
            ax = axs[i_dataset, nr - 1]
            if nr > 1:
                chunk_edge = X[inds['index'][start]][split_dbscan.split_dim]
                eps_edge = chunk_edge - eps# * params["min_samples"]
                eps_start = np.searchsorted(X[:, 0], eps_edge, sorter = inds['index'], side='right')
                chunk_index = inds['index'][eps_start:end+1]
            else:
                chunk_index = inds['index'][start:end+1]
            #chunk = X[inds['index'][start:end]]
            dbscan.fit(X[chunk_index])
            labels = dbscan.labels_
            split_dbscan.labels_ = labels
            active_mask = split_dbscan.active_mask(X[chunk_index])
            if (nr > 1) and (eps_start != start):
                eps_old = inds[eps_start:start]
                temp = np.zeros((eps_old.shape[0],), [('index', int), ('labels', int)])
                temp['index'] = chunk_index[:eps_old.shape[0]]
                temp['labels'] = labels[:eps_old.shape[0]]
                #eps_new = temp[eps_old['index']]
                merged = rf.join_by('index', temp, eps_old, 'inner', usemask=False)
                if len(merged) > 0:
                    # merged = merged[merged['labels1'] != merged['labels2']]
                    # step 1: Drop -1 in relabeled dataset (so that mapping from 
                    # earlier cluster member to no cluster membership in current
                    # chunk is not performed for any of the points in the overlap).
                    merged = merged[merged['labels2'] != -1]
                    # step 1: Remap points in the overlap that have changed from -1
                    # to cluster membership.
                    selected = merged[merged['labels1'] == -1]

                    mask_to_1s = np.isin(inds['index'][eps_start:end+1], selected['index'])
                    #inds['labels'][ind] = selected['labels2']

                    merged = merged[merged['labels1'] != -1]

                    if len(merged ) > 0:
                        mapping = np.unique(merged[['labels1', 'labels2']])
                        keys = np.array([x[0] for x in mapping])
                        values = np.array([x[1] for x in mapping])
                        # check for repeated keys and if present remove from keys/values
                        # and relabel afected clusters in the previous chunks.
                        # set length check is quicker for small arrays but np.unique
                        # would be still needed if True, so skipping for now. 
                        # if len(set(keys)) != len(keys):
                        vals, ind_start, count = np.unique(keys, return_counts=True, return_index=True)
                        if np.any(count > 1):
                            # relabel newly merged clusters. Currenty nested python loops,
                            # perhaps could be vectorised if slow for large number of clusters.
                            for repeat_ind in ind_start[count > 1]:
                                r_values = values[repeat_ind:repeat_ind+count[repeat_ind]]
                                for rep_value in r_values[1:]:
                                    inds['labels'][inds['labels'] == rep_value] = r_values[0]
                                    values[values == rep_value] = r_values[0] 
                            # keep only unique 
                            keys = keys[ind_start]
                            values = values[ind_start]
                        # sort_idx = np.argsort(keys)
                        idx = np.searchsorted(keys, labels)

                        idx[idx==len(values)] = 0
                        mask = keys[idx] == labels
                        labels = np.where(mask, values[idx], labels)
                        if not np.all(mask):
                            labels[~mask] = normalize_labels(labels[~mask], inds['labels'].max())
                        # check for repeated keys and relabel merged clustersn
                    else:
                        labels = normalize_labels(labels, inds['labels'].max())
                    inds['labels'][eps_start:end+1][~mask_to_1s] = labels[~mask_to_1s]
                else:
                    inds['labels'][start:end+1] = labels
            else:
                inds['labels'][start:end+1] = labels
            #y_pred = labels_complete
            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        max(int(max(inds['labels']) + 1), 0),
                    )
                )
            )
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            ax.scatter(X[inds['index'][:end+1]][:, 0], X[inds['index'][:end+1]][:, 1],
                    s=10, color=colors[inds['labels'][:end+1]])
            #ax.scatter(X[chunk_index][:, 0], X[chunk_index][:, 1], s=40, marker='o', color='grey', facecolors='none')
            # plt.scatter(chunk

            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xticks(())
            ax.set_yticks(())
            edge = X[chunk_index[-1]][0]
            if ('eps_start' in globals()) and (nr > 1) :
                ax.axvline(x = X[inds['index'][eps_start]][0], color = 'grey', linestyle='--', label = 'edge')
            ax.axvline(x = X[inds['index'][start]][0], color = 'grey', label = 'edge')
            ax.axvline(x = edge, color = 'grey', label = 'edge')
            if i_dataset == 0:
                ax.set_title(f'splitDBSCAN chunk = {nr}')

        ax = axs[i_dataset, -1]

        dbscan.fit(X)
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    max(int(max(dbscan.labels_) + 1), 0),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax.scatter(X[:, 0], X[:, 1], s=10, color=colors[dbscan.labels_])
        #ax.scatter(X[chunk_index][:, 0], X[chunk_index][:, 1], s=40, marker='o', color='grey', facecolors='none')
        # plt.scatter(chunk

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_yticks(())
        if i_dataset == 0:
            ax.set_title('vanila DBSCAN')
        print('No unique: ', len(np.unique(inds['labels'])), len(np.unique(dbscan.labels_)))
        print('No -1: ', len(inds[inds['labels']==-1]), len(dbscan.labels_[dbscan.labels_==-1]))
    plt.savefig('test.png', bbox_inches='tight', dpi=80)
    plt.show()
