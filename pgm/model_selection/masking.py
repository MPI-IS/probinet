"""
This module provides functions for shuffling indices and extracting masks for selecting the held-out set in the adjacency tensor and design matrix.
"""

from typing import List, Optional, Tuple

import numpy as np


def shuffle_indices(N: int, L: int, rseed: int) -> List[np.ndarray]:
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
    List[np.ndarray]
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


def shuffle_indicesG(N: int, L: int, rseed: int = 10) -> List[List[Tuple[int, int]]]:
    """
    Extract a maskG using KFold.

    Parameters
    ----------
    N : int
        Number of nodes.
    L : int
        Number of layers.
    rseed : int, optional
        Random seed, by default 10.

    Returns
    -------
    List[List[Tuple[int, int]]]
        Shuffled indices for each layer.
    """
    # Create a random number generator with the specified random seed
    rng = np.random.RandomState(rseed)

    # Generate indices for each layer using list comprehension
    idxG = [[(i, j) for i in range(N) for j in range(N)] for _ in range(L)]

    # Shuffle indices for each layer
    for l in range(L):
        rng.shuffle(idxG[l])

    return idxG


def shuffle_indicesX(N: int, rseed: int = 10) -> np.ndarray:
    """
    Extract a maskX using KFold.

    Parameters
    ----------
    N : int
        Number of nodes.
    rseed : int, optional
        Random seed, by default 10.

    Returns
    -------
    np.ndarray
        Shuffled indices.
    """
    # Create a random number generator with the specified random seed
    rng = np.random.RandomState(rseed)
    idxX = np.arange(N)

    # Shuffle the indices
    rng.shuffle(idxX)

    return idxX


def extract_masks(
    N: int,
    L: int,
    idxG: Optional[List[List[Tuple[int, int]]]] = None,
    idxX: Optional[List[int]] = None,
    cv_type: str = "kfold",
    NFold: int = 5,
    fold: int = 0,
    rseed: int = 10,
    out_mask: bool = False,
    out_folder: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the masks for selecting the held out set in the adjacency tensor and design matrix.

    Parameters
    ----------
    N : int
        Number of nodes.
    L : int
        Number of layers.
    idxG : Optional[List[List[Tuple[int, int]]]], optional
        Each list has the indexes of the entries of the adjacency matrix of layer L, when cv is set to kfold.
    idxX : Optional[List[int]], optional
        List with the indexes of the entries of design matrix, when cv is set to kfold.
    cv_type : str, optional
        Type of cross-validation: kfold or random, by default 'kfold'.
    NFold : int, optional
        Number of C-fold, by default 5.
    fold : int, optional
        Current fold, by default 0.
    rseed : int, optional
        Random seed, by default 10.
    out_mask : bool, optional
        If set to True, the masks are saved into files, by default False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Mask for selecting the held out set in the adjacency tensor and design matrix.
    """
    # Initialize masks
    maskG = np.zeros((L, N, N), dtype=bool)
    maskX = np.zeros(N, dtype=bool)

    if cv_type == "kfold":  # sequential order of folds
        # adjacency tensor
        assert L == len(idxG)
        for l in range(L):
            n_samples = len(idxG[l])
            test = idxG[l][
                fold * (n_samples // NFold) : (fold + 1) * (n_samples // NFold)
            ]
            for idx in test:
                maskG[l][idx] = 1

        # design matrix
        testcov = idxX[fold * (N // NFold) : (fold + 1) * (N // NFold)]
        maskX[testcov] = 1

    else:  # random split for choosing the test set
        rng = np.random.RandomState(rseed)  # Mersenne-Twister random number generator
        maskG = rng.binomial(1, 1.0 / float(NFold), size=(L, N, N))
        maskX = rng.binomial(1, 1.0 / float(NFold), size=N)

    return maskG, maskX


def extract_mask_kfold(
    indices: List[np.ndarray], N: int, fold: int = 0, NFold: int = 5
) -> np.ndarray:
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
        test = indices[l][
            fold * (n_samples // NFold) : (fold + 1) * (n_samples // NFold)
        ]

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
