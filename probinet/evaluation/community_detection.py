"""
Functions for evaluating community detection.
"""

from contextlib import suppress
from typing import Set

import numpy as np


def calculate_metric(ground_truth: Set[int], detected: Set[int], metric: str) -> float:
    """
    Calculates a performance metric based on the similarity between the ground truth
    set and the detected set. Supports "f1" and "jaccard" metrics. It is typically
    used for evaluating the quality of prediction sets in relation to the true
    reference sets.

    Parameters
    ----------
    ground_truth
        The set of true (ground truth) items.
    detected
        The set of predicted (detected) items.
    metric : str
        The metric to calculate. Must be either "f1" for the F1 score or "jaccard"
        for the Jaccard index.

    Returns
    -------
    float
        The calculated metric value as a float. Returns 0.0 if there is no overlap
        between the ground truth and the detected sets.

    Raises
    ------
    ValueError
        If the provided metric is unsupported or invalid.

    Notes
    -----
    - The F1 score is calculated as the harmonic mean of precision and recall.
    - The Jaccard index is calculated as the ratio of the size of the intersection
      of the sets to the size of their union.
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
    Compute a similarity metric to evaluate the quality of inferred community memberships.

    This function compares inferred community memberships (`U_infer`) against the ground truth (`U0`)
    using supported performance metrics like F1-score or Jaccard Index. The inferred and true
    community memberships are represented as **membership matrices** (`U_infer` and `U0`) where
    each row corresponds to a node, and columns represent its membership strength across K communities.

    Parameters
    ----------
    U_infer
        Inferred community membership matrix (shape: [N, K]), where N is the number
        of nodes and K is the number of communities. This matrix is created during
        the community detection process.

    U0
        Ground truth membership matrix (shape: [N, K]) that represents the true community
        membership strengths of nodes.

    metric
        The similarity metric to use for evaluation, either:
        - "f1": F1-score (harmonic mean of precision and recall for membership overlap).
        - "jaccard": Jaccard index (intersection-over-union for memberships).

    com
        If True, interpret each row of `U_infer` directly as community members.
        Otherwise, threshold each row to determine community assignment.

    Returns
    -------
    float
        A similarity score based on the chosen metric. The value is between 0 and 1,
        where higher values indicate better alignment between the inferred and true
        community structures.

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
    Computes a permutation matrix that aligns the inferred and ground truth community memberships.

    This function calculates a permutation matrix to align two membership matrices, `U_infer` (inferred community
    structure) and `U0` (ground truth community structure). The permutation ensures that columns of `U_infer`
    are reordered to best match those of `U0`. It helps resolve discrepancies caused by arbitrary ordering
    of communities across different runs or datasets.

    Parameters
    ----------
    U_infer
        Inferred community membership matrix (shape: [N, K]), where N is the number of nodes and K is the number
        of communities. Each row represents the membership strengths of a node across K inferred communities.

    U0
        Ground truth community membership matrix (shape: [N, K]), where each row represents the true membership
        strengths of a node across K communities.

    Returns
    -------
    np.ndarray
        A permutation matrix (shape: [K, K]) where each row and column contains a single `1`. The matrix reorders
        the columns of `U_infer` to maximize alignment with the columns of `U0`.

    Notes
    -----
    - The membership matrices (`U_infer` and `U0`) represent soft clustering of nodes into communities, where
      each row indicates the strength of a node's association with each community.
    - Permutation is necessary because community detection algorithms often produce outputs where the community
      labels have no strict correspondence (e.g., community 1 in `U_infer` may correspond to community 3 in `U0`).

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
    Compute the cosine similarity between inferred and ground truth community memberships after alignment.

    This function calculates the average cosine similarity between two community membership matrices, `U_infer`
    (inferred communities) and `U0` (ground truth communities). Before computing similarity, the function aligns
    the columns of `U_infer` with `U0` using a permutation matrix to ensure consistency in community ordering.

    Cosine similarity measures the angular similarity between vectors, making it a suitable metric for comparing
    normalized membership strengths of nodes in a soft clustering context.

    Parameters
    ----------
    U_infer
        Inferred community membership matrix (shape: [N, K]), where N is the number of nodes and K is the number
        of communities. Each row represents the degree to which a node belongs to each community.

    U0
        Ground truth community membership matrix (shape: [N, K]), where each row represents the true membership
        strengths of nodes across K communities.

    Returns
    -------
    tuple
        A tuple containing:
        - `np.ndarray`: The aligned inferred membership matrix (`U_infer`) after applying the permutation matrix.
        - `float`: The average cosine similarity score, a measure of the alignment between `U_infer` and `U0`,
          where higher values indicate stronger similarity.

    Notes
    -----
    - Cosine similarity ranges between -1 and 1, but in this context (non-negative memberships), it typically
      falls between 0 and 1.
    - `U_infer` is first aligned to `U0` using a permutation matrix to resolve potential column order mismatches
      in community labels.
    - Normalization is applied to each row of `U_infer` and `U0` before computing the similarity, ensuring equal
      weight across nodes.

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
