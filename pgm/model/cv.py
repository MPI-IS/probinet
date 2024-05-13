"""
It provides functions for cross-validation.
"""
import sys
from typing import List, Optional

import numpy as np

from pgm.input.tools import transpose_ij3
from pgm.output.evaluate import calculate_Z, lambda_full


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

# The functions below are not used in the current implementation. They are taken from DynCRep,
# not sure if it would be needed at some point; if so, then probably in cv functions


def cosine_similarity(U_infer, U0):
    """
    It is assumed that matrices are row-normalized
    """
    P = CalculatePermutation(U_infer, U0)
    U_infer = np.dot(U_infer, P)  # Permute infered matrix
    N, K = U0.shape
    U_infer0 = U_infer.copy()
    U0tmp = U0.copy()
    cosine_sim = 0.
    norm_inf = np.linalg.norm(U_infer, axis=1)
    norm0 = np.linalg.norm(U0, axis=1)
    for i in range(N):
        if (norm_inf[i] > 0.):
            U_infer[i, :] = U_infer[i, :] / norm_inf[i]
        if (norm0[i] > 0.):
            U0[i, :] = U0[i, :] / norm0[i]

    for k in range(K):
        cosine_sim += np.dot(np.transpose(U_infer[:, k]), U0[:, k])
    U0 = U0tmp.copy()
    return U_infer0, cosine_sim / float(N)


def CalculatePermutation(U_infer, U0):
    """
    Permuting the overlap matrix so that the groups from the two partitions correspond
    U0 has dimension NxK, reference memebership
    """
    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)  # dim=RANKxRANK
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # Permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining submatrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.
        c_index = 1
        r_index = 1
        for i in range(RANK):
            if columns[i] == 0:
                for j in range(RANK):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i]
                            c_index = i
                            r_index = j

        P[r_index, c_index] = 1
        columns[c_index] = 1
        rows[r_index] = 1

    return P


def Likelihood_conditional(M, beta, data, data_tm1, EPS=1e-12):
    """
        Compute the log-likelihood of the data conditioned in the previous time step

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T : sptensor/dtensor
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

        Returns
        -------
        l : float
             log-likelihood value.
    """
    l = - M.sum()
    sub_nz_and = np.logical_and(data > 0, (1 - data_tm1) > 0)
    Alog = data[sub_nz_and] * (1 - data_tm1)[sub_nz_and] * np.log(M[sub_nz_and] + EPS)
    l += Alog.sum()
    sub_nz_and = np.logical_and(data > 0, data_tm1 > 0)
    l += np.log(1 - beta + EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    sub_nz_and = np.logical_and(data_tm1 > 0, (1 - data) > 0)
    l += np.log(beta + EPS) * ((1 - data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    if np.isnan(l):
        print("Likelihood is NaN!!!!")
        sys.exit(1)
    else:
        return l


def probabilities(
        structure: str,
        sizes: List[int],
        N: int = 100,
        K: int = 2,
        avg_degree: float = 4.,
        alpha: float = 0.1,
        beta: Optional[float] = None) -> np.ndarray:
    """
    Return the CxC array with probabilities between and within groups.

    Parameters
    ----------
    structure : str
                Structure of the layer, e.g. assortative, disassortative, core-periphery or directed-biased.
    sizes : List[int]
            List with the sizes of blocks.
    N : int
        Number of nodes.
    K : int
        Number of communities.
    avg_degree : float
                 Average degree over the nodes.
    alpha : float
            Alpha value. Default is 0.1.
    beta : float
           Beta value. Default is 0.3 * alpha.

    Returns
    -------
    p : np.ndarray
        Ar
    """
    if beta is None:
        beta = alpha * 0.3
    p1 = avg_degree * K / N
    if structure == 'assortative':
        p = p1 * alpha * np.ones((len(sizes), len(sizes)))  # secondary-probabilities
        np.fill_diagonal(p, p1)  # primary-probabilities
    elif structure == 'disassortative':
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(p, alpha * p1)
    elif structure == 'core-periphery':
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(np.fliplr(p), alpha * p1)
        p[1, 1] = beta * p1
    elif structure == 'directed-biased':
        p = alpha * p1 * np.ones((len(sizes), len(sizes)))
        p[0, 1] = p1
        p[1, 0] = beta * p1

    return p
