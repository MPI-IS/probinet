"""
Functions for evaluating community detection.
"""

from contextlib import suppress

import numpy as np


def compute_community_detection_metric(
    U_infer: np.ndarray, U0: np.ndarray, metric: str = "f1", com: bool = False
) -> float:
    """
    Compute an evaluation metric for community detection.
    """
    if metric not in {"f1", "jaccard"}:
        raise ValueError('The similarity measure can be either "f1" or "jaccard"!')
    K = U0.shape[1]
    gt = {}
    d = {}
    threshold = 1 / U0.shape[1]
    for i in range(K):
        gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
        if com:
            with suppress(IndexError):
                d[i] = U_infer[i]
        else:
            d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
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
