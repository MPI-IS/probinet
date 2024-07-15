"""
It provides functions for cross-validation.
"""

from typing import List

import numpy as np

from pgm.input.tools import log_and_raise_error


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


# The functions below are not used in the current implementation. They are taken from DynCRep,
# not sure if it would be needed at some point; if so, then probably in cv functions


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
    l = -M.sum()
    sub_nz_and = np.logical_and(data > 0, (1 - data_tm1) > 0)
    Alog = data[sub_nz_and] * (1 - data_tm1)[sub_nz_and] * np.log(M[sub_nz_and] + EPS)
    l += Alog.sum()
    sub_nz_and = np.logical_and(data > 0, data_tm1 > 0)
    l += np.log(1 - beta + EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    sub_nz_and = np.logical_and(data_tm1 > 0, (1 - data) > 0)
    l += np.log(beta + EPS) * ((1 - data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    if np.isnan(l):
        log_and_raise_error(ValueError, "Likelihood is NaN!")
    return l


def evalu(U_infer, U0, metric="f1", com=False):
    """
    Compute an evaluation metric.

    Compare a set of ground-truth communities to a set of detected communities. It matches every detected
    community with its most similar ground-truth community and given this matching, it computes the performance;
    then every ground-truth community is matched with a detected community and again computed the performance.
    The final performance is the average of these two metrics.

    Parameters
    ----------
    U_infer : ndarray
              Inferred membership matrix (detected communities).
    U0 : ndarray
         Ground-truth membership matrix (ground-truth communities).
    metric : str
             Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
             if 'jaccard', it uses the Jaccard similarity.
    com : bool
          Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
          membership matrix (False).

    Returns
    -------
    Evaluation metric.
    """

    if metric not in {"f1", "jaccard"}:
        raise ValueError(
            'The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
            "Jaccard similarity!"
        )

    K = U0.shape[1]

    gt = {}
    d = {}
    threshold = 1 / U0.shape[1]
    for i in range(K):
        gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
        if com:
            try:
                d[i] = U_infer[i]
            except IndexError:
                pass
        else:
            d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
    # First term
    R = 0
    for i in np.arange(K):
        ground_truth = set(gt[i])
        _max = -1
        M = 0
        for j in d.keys():
            detected = set(d[j])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == "f1":
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == "jaccard":
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        R += _max
    # Second term
    S = 0
    for j in d.keys():
        detected = set(d[j])
        _max = -1
        M = 0
        for i in np.arange(K):
            ground_truth = set(gt[i])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == "f1":
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == "jaccard":
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        S += _max

    return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)


def evalu(U_infer, U0, metric="f1", com=False):
    """
    Compute an evaluation metric.

    Compare a set of ground-truth communities to a set of detected communities. It matches every detected
    community with its most similar ground-truth community and given this matching, it computes the performance;
    then every ground-truth community is matched with a detected community and again computed the performance.
    The final performance is the average of these two metrics.

    Parameters
    ----------
    U_infer : ndarray
              Inferred membership matrix (detected communities).
    U0 : ndarray
         Ground-truth membership matrix (ground-truth communities).
    metric : str
             Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
             if 'jaccard', it uses the Jaccard similarity.
    com : bool
          Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
          membership matrix (False).

    Returns
    -------
    Evaluation metric.
    """

    if metric not in {"f1", "jaccard"}:
        raise ValueError(
            'The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
            "Jaccard similarity!"
        )

    K = U0.shape[1]

    gt = {}
    d = {}
    threshold = 1 / U0.shape[1]
    for i in range(K):
        gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
        if com:
            try:
                d[i] = U_infer[i]
            except:
                pass
        else:
            d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
    # First term
    R = 0
    for i in np.arange(K):
        ground_truth = set(gt[i])
        _max = -1
        M = 0
        for j in d.keys():
            detected = set(d[j])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == "f1":
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == "jaccard":
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        R += _max
    # Second term
    S = 0
    for j in d.keys():
        detected = set(d[j])
        _max = -1
        M = 0
        for i in np.arange(K):
            ground_truth = set(gt[i])
            if len(ground_truth & detected) != 0:
                precision = len(ground_truth & detected) / len(detected)
                recall = len(ground_truth & detected) / len(ground_truth)
                if metric == "f1":
                    M = 2 * (precision * recall) / (precision + recall)
                elif metric == "jaccard":
                    M = len(ground_truth & detected) / len(ground_truth.union(detected))
            if M > _max:
                _max = M
        S += _max

    return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)
