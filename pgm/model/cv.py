"""
It provides functions for cross-validation.
"""
from typing import List

import numpy as np


def extract_mask_kfold(indices: List[np.ndarray], N: int, fold: int = 0,
                       NFold: int = 5) -> np.ndarray:
    """
    Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but
    possibly not (j,i).
    KFold means no train/test sets intersect across the K folds.

    Parameters
    ----------
    indices : List[np.ndarray]
              Indices of the adjacency tensor in a shuffled order.
    N : int
        Number of nodes.
    fold : int
           Current fold.
    NFold : int
            Number of total folds.

    Returns
    -------
    mask : np.ndarray
           Mask for selecting the held out set in the adjacency tensor.
    """

    # Get the number of layers
    L = len(indices)

    # Initialize an empty boolean mask with dimensions (L, N, N)
    mask = np.zeros((L, N, N), dtype=bool)

    # Loop over each layer
    for l in range(L):
        # Get the number of samples in the current layer
        n_samples = len(indices[l])

        # Determine the test set indices for the current fold
        test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]

        # Create a boolean mask for the test set
        mask0 = np.zeros(n_samples, dtype=bool)
        mask0[test] = 1

        # Reshape the mask to match the dimensions (N, N) and store it in the result mask array
        mask[l] = mask0.reshape((N, N))

    # Return the final mask
    return mask


def shuffle_indices_all_matrix(N: int, L: int, rseed: int = 10) -> List[np.ndarray]:
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
    indices : List[np.ndarray]
              Indices in a shuffled order.
    """

    # Calculate the total number of samples in the adjacency tensor
    n_samples = int(N * N)

    # Create a list of arrays, where each array contains the range of indices for a layer
    indices = [np.arange(n_samples) for _ in range(L)]

    # Create a random number generator with the specified random seed
    rng = np.random.default_rng(rseed)

    # Loop over each layer and shuffle the corresponding indices
    for l in range(L):
        rng.shuffle(indices[l])

    # Return the shuffled indices
    return indices
