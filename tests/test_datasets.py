import numpy as np

from sklearn import datasets


def sklearn_datasets():
    """Return sklearn example datasets with configuration parameters"""
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
                "eps": 0.3,
                "min_samples": 7,
            },
        ),
        (
            noisy_moons,
            {
                "eps": 0.3,
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
        (blobs, {"eps": 0.3, "min_samples": 7}),
        (no_structure, {"eps": 0.3, "min_samples": 7}),
    ]
    return test_datasets
