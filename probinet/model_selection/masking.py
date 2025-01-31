"""
This module provides functions for shuffling indices and extracting masks for selecting the held-out set in the adjacency tensor and design matrix.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..utils.tools import get_or_create_rng


def shuffle_indices(
    N: int, L: int, rng: Optional[np.random.Generator] = None
) -> (List)[np.ndarray]:
    """
    Shuffle the indices of the adjacency tensor.

    Parameters
    ----------
    N
        Number of nodes.
    L
        Number of layers.
    rng
        Random number generator.
    Returns
    -------
    Indices in a shuffled order.
        Indices in a shuffled order.
    """
    # Calculate the total number of samples in the adjacency tensor
    n_samples = int(N * N)

    # Create a list of arrays, where each array contains the range of indices for a layer
    indices = [np.arange(n_samples) for _ in range(L)]

    # Create a random number generator with the specified random seed
    rng = get_or_create_rng(rng)

    # Loop over each layer and shuffle the corresponding indices
    for l in range(L):
        rng.shuffle(indices[l])

    # Return the shuffled indices
    return indices


def shuffle_indicesG(
    N: int, L: int, rng: Optional[np.random.Generator] = None
) -> List[List[Tuple[int, int]]]:
    """
    Parameters
    ----------
    N
        Number of nodes.
    L
        Number of layers.
    rng
        Random number generator.
    Returns
    -------
    Shuffled indices for each layer.
    """

    # Create a random number generator with the specified random seed
    rng = get_or_create_rng(rng)

    # Generate indices for each layer using list comprehension
    idxG = [[(i, j) for i in range(N) for j in range(N)] for _ in range(L)]

    # Shuffle indices for each layer
    for layer in range(L):
        rng.shuffle(idxG[layer])

    return idxG


def shuffle_indicesX(N: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Extract a maskX using KFold.

    Parameters
    ----------
    N
        Number of nodes.
    rng
        Random number generator.
    Returns
    -------
    Shuffled indices.
    """
    # Create a random number generator with the specified random seed
    rng = get_or_create_rng(rng)
    idxX = np.arange(N)

    # Shuffle the indices
    rng.shuffle(idxX)

    return idxX


def extract_masks(
    N: int,
    L: int,
    idxG: list[list[Tuple[int, int]]],
    idxX: Sequence[int],
    cv_type: str = "kfold",
    NFold: int = 5,
    fold: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the masks for selecting the held out set in the adjacency tensor and design matrix.

    Parameters
    ----------
    N
        Number of nodes.
    L
        Number of layers.
    idxG
        Each list has the indexes of the entries of the adjacency matrix of layer L, when cv is set to kfold.
    idxX
        List with the indexes of the entries of design matrix, when cv is set to kfold.
    cv_type
        Type of cross-validation: kfold or random, by default 'kfold'.
    NFold
        Number of C-fold, by default 5.
    fold
        Current fold, by default 0.
    rng
        Random number generator.
    Returns
    -------
    Mask for selecting the held out set in the adjacency tensor and design matrix.
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
        rng = get_or_create_rng(rng)
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
    indices
              Indices of the adjacency tensor in a shuffled order.
    N
        Number of nodes.
    fold
           Current fold.
    NFold
            Number of total folds.
    Returns
    -------
    mask
           Mask for selecting the held out set in the adjacency tensor. It is made of 0s and 1s,
           where 1s represent that the element (i,j) should be used.
           where 1s represent that the element (i,j) should be used.
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


def shuffle_indices_all_matrix(
    N: int, L: int, rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Shuffle the indices of the adjacency tensor.

    Parameters
    ----------
    N
        Number of nodes.
    L
        Number of layers.
    rng
            Random number generator.
    Returns
    -------
    indices
              Indices in a shuffled order.
              Indices in a shuffled order.
    """

    # Create a random number generator with the specified random seed
    rng = get_or_create_rng(rng)

    # Calculate the total number of samples in the adjacency tensor
    n_samples = int(N * N)

    # Create a list of arrays, where each array contains the range of indices for a layer
    indices = [np.arange(n_samples) for _ in range(L)]

    # Loop over each layer and shuffle the corresponding indices
    for layer in range(L):
        rng.shuffle(indices[layer])

    # Return the shuffled indices
    return indices
