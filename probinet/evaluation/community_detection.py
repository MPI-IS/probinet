"""
Functions for evaluating community detection.
"""

from contextlib import suppress
from typing import Set

import numpy as np


def calculate_metric(ground_truth: Set[int], detected: Set[int], metric: str) -> float:
    """
    Calculate a metric for evaluating community detection.

    Parameters
    ----------
    ground_truth : Set[int]
        The set of ground truth nodes.
    detected : Set[int]
        The set of detected nodes.
    metric : str
        The metric to use for evaluation ('f1' or 'jaccard').

    Returns
    -------
    float
        The calculated metric value.
    """
    if not len(ground_truth.intersection(detected)):
        return 0.0
    precision = len(ground_truth.intersection(detected)) / len(detected)
    recall = len(ground_truth.intersection(detected)) / len(ground_truth)
    if metric == "f1":
        return 2 * (precision * recall) / (precision + recall)
    elif metric == "jaccard":
        return len(ground_truth.intersection(detected)) / len(
            ground_truth.union(detected)
        )


def compute_community_detection_metric(
    U_infer: np.ndarray, U0: np.ndarray, metric: str = "f1", com: bool = False
) -> float:
    """
    Compute an evaluation metric for community detection.
    """
    if metric not in {"f1", "jaccard"}:
        raise ValueError('The similarity measure can be either "f1" or "jaccard"!')

    K = U0.shape[1]
    threshold = 1 / K
    # Create the ground truth dictionary for each community in the original partition. The key is
    # the community index and the value is the set of nodes in that community, i.e., the nodes
    # with a value greater than the threshold in the corresponding column of the original partition.
    gt = {i: set(np.argwhere(U0[:, i] > threshold).flatten()) for i in range(K)}
    d = {}
    for i in range(K):
        if com:
            with suppress(IndexError):
                d[i] = set(U_infer[i])
        else:
            d[i] = set(np.argwhere(U_infer[:, i] > threshold).flatten())

    R = sum(
        max(calculate_metric(gt[i], d[j], metric) for j in d.keys()) for i in range(K)
    )
    S = sum(
        max(calculate_metric(gt[i], d[j], metric) for i in range(K)) for j in d.keys()
    )
    # Return the average of the two measures
    return np.round(R / (2 * K) + S / (2 * len(d)), 4)


def compute_permutation_matrix(U_infer: np.ndarray, U0: np.ndarray) -> np.ndarray:
    """
    Permute the overlap matrix so that the groups from the two partitions correspond.
    """
    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))
    for _ in range(RANK):
        max_entry = 0.0
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


def cosine_similarity(U_infer: np.ndarray, U0: np.ndarray) -> tuple:
    """
    Compute the cosine similarity between two row-normalized matrices.
    """
    P = compute_permutation_matrix(U_infer, U0)
    U_infer = np.dot(U_infer, P)
    N, K = U0.shape
    U_infer0 = U_infer.copy()
    U0tmp = U0.copy()
    cosine_sim = 0.0
    norm_inf = np.linalg.norm(U_infer, axis=1)
    norm0 = np.linalg.norm(U0, axis=1)
    for i in range(N):
        if norm_inf[i] > 0.0:
            U_infer[i, :] = U_infer[i, :] / norm_inf[i]
        if norm0[i] > 0.0:
            U0[i, :] = U0[i, :] / norm0[i]
    for k in range(K):
        cosine_sim += np.dot(np.transpose(U_infer[:, k]), U0[:, k])
    U0 = U0tmp.copy()
    return U_infer0, cosine_sim / float(N)
