"""
This module provides functions for evaluating the accuracy of attribute predictions and computing evaluation metrics for community detection.
"""

from contextlib import suppress
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import metrics

from pgm.model_selection.labeling import extract_true_label, predict_label


def covariates_accuracy(
    X: pd.DataFrame,
    u: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Return the accuracy of the attribute prediction, computed as the fraction of correctly classified examples.

    Parameters
    ----------
    X : pd.DataFrame
        Pandas DataFrame object representing the one-hot encoding version of the design matrix.
    u : np.ndarray
        Membership matrix (out-degree).
    v : np.ndarray
        Membership matrix (in-degree).
    beta : np.ndarray
        Beta parameter matrix.
    mask : Optional[np.ndarray]
        Mask for selecting a subset of the design matrix.

    Returns
    -------
    float
        Fraction of correctly classified examples.
    """
    # Extract true labels from the design matrix
    true_label = extract_true_label(X, mask=mask)

    # Compute predicted labels
    pred_label = predict_label(X, u, v, beta, mask=mask)

    # Calculate accuracy score
    acc = metrics.accuracy_score(true_label, pred_label)

    return acc


def evalu(
    U_infer: np.ndarray, U0: np.ndarray, metric: str = "f1", com: bool = False
) -> float:
    """
    Compute an evaluation metric.

    Compare a set of ground-truth communities to a set of detected communities. It matches every detected
    community with its most similar ground-truth community and given this matching, it computes the performance;
    then every ground-truth community is matched with a detected community and again computed the performance.
    The final performance is the average of these two metrics.

    Parameters
    ----------
    U_infer : np.ndarray
        Inferred membership matrix (detected communities).
    U0 : np.ndarray
        Ground-truth membership matrix (ground-truth communities).
    metric : str
        Similarity measure between the true community and the detected one. If 'f1', it uses the F1-score,
        if 'jaccard', it uses the Jaccard similarity.
    com : bool
        Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
        membership matrix (False).

    Returns
    -------
    float
        Evaluation metric.
    """
    # Validate the metric parameter
    if metric not in {"f1", "jaccard"}:
        raise ValueError(
            'The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
            "Jaccard similarity!"
        )

    # Number of communities
    K = U0.shape[1]

    # Initialize dictionaries for ground-truth and detected communities
    gt = {}
    d = {}
    threshold = 1 / U0.shape[1]

    # Populate ground-truth communities
    for i in range(K):
        gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
        if com:
            with suppress(IndexError):
                d[i] = U_infer[i]
        else:
            d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())

    # First term: match detected communities to ground-truth communities
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

    # Second term: match ground-truth communities to detected communities
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

    # Return the average of the two metrics
    return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)
