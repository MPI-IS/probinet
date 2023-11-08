"""
It provides functions for cross-validation.
"""
import numpy as np


def extract_mask_kfold(indices, N, fold=0, NFold=5):
    """
        Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
        KFold means no train/test sets intersect across the K folds.

        Parameters
        ----------
        indices : ndarray
                  Indices of the adjacency tensor in a shuffled order.
        N : int
            Number of nodes.
        fold : int
               Current fold.
        NFold : int
                Number of total folds.

        Returns
        -------
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor.
    """

    L = len(indices)
    mask = np.zeros((L, N, N), dtype=bool)
    for l in range(L):
        n_samples = len(indices[l])
        test = indices[l][fold * (n_samples // NFold):(fold + 1) *
                          (n_samples // NFold)]
        mask0 = np.zeros(n_samples, dtype=bool)
        mask0[test] = 1
        mask[l] = mask0.reshape((N, N))

    return mask


def shuffle_indices_all_matrix(N, L, rseed=10):
    """
        Shuffle the indices of the adjacency tensor.

        Parameters
        ----------
        N : int
            Number of nodes.
        L : int
            Number of layers.
        rseed : int
                Random seed.

        Returns
        -------
        indices : ndarray
                  Indices in a shuffled order.
    """

    n_samples = int(N * N)
    indices = [np.arange(n_samples) for _ in range(L)]
    rng = np.random.RandomState(rseed)
    for l in range(L):
        rng.shuffle(indices[l])

    return indices
