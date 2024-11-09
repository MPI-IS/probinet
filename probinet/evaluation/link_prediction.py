"""
Functions for evaluating link prediction.
"""

from typing import Optional

import numpy as np
from sklearn import metrics

from probinet.evaluation.expectation_computation import (
    compute_expected_adjacency_tensor_multilayer,
)


def compute_link_prediction_AUC(
    data0: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the AUC (Area Under the Curve) for link prediction.

    Parameters
    ----------
    data0 : np.ndarray
        The original adjacency matrix.
    pred : np.ndarray
        The predicted adjacency matrix.
    mask : Optional[np.ndarray], optional
        The mask to apply on the data, by default None.

    Returns
    -------
    float
        The AUC value for the link prediction.
    """
    data = (data0 > 0).astype("int")
    if mask is None:
        fpr, tpr, _ = metrics.roc_curve(data.flatten(), pred.flatten())
    else:
        fpr, tpr, _ = metrics.roc_curve(data[mask > 0], pred[mask > 0])
    return metrics.auc(fpr, tpr)


def compute_multilayer_link_prediction_AUC(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the AUC (Area Under the Curve) for link prediction in multilayer data.

    Parameters
    ----------
    B : np.ndarray
        The original adjacency tensor.
    u : np.ndarray
        The first factor matrix.
    v : np.ndarray
        The second factor matrix.
    w : np.ndarray
        The third factor matrix.
    mask : Optional[np.ndarray], optional
        The mask to apply on the data, by default None.

    Returns
    -------
    float
        The AUC value for the link prediction in multilayer data.
    """
    expected_adjacency = compute_expected_adjacency_tensor_multilayer(u, v, w)
    if mask is None:
        ranked_predictions = list(zip(expected_adjacency.flatten(), B.flatten()))
        num_positive_samples = B.sum()
    else:
        ranked_predictions = list(zip(expected_adjacency[mask > 0], B[mask > 0]))
        num_positive_samples = B[mask > 0].sum()
    ranked_predictions.sort(key=lambda x: x[0], reverse=False)
    total_samples = len(ranked_predictions)
    num_negative_samples = total_samples - num_positive_samples
    return compute_AUC_from_ranked_predictions(
        ranked_predictions, num_positive_samples, num_negative_samples
    )


def compute_AUC_from_ranked_predictions(
    ranked_predictions: list[tuple[float, int]],
    num_positive_samples: int,
    num_negative_samples: int,
) -> float:
    """
    Calculate the AUC (Area Under the Curve) for the given ranked list of predictions.

    Parameters
    ----------
    ranked_predictions : list[tuple[float, int]]
        The ranked list of predictions, where each tuple contains a score and the actual value.
    num_positive_samples : int
        The number of positive samples.
    num_negative_samples : int
        The number of negative samples.

    Returns
    -------
    float
        The AUC value for the ranked predictions.
    """
    y = 0.0
    bad = 0.0
    for score, actual in ranked_predictions:
        if actual > 0:
            y += 1
        else:
            bad += y
    AUC = 1.0 - (bad / (num_positive_samples * num_negative_samples))
    return AUC
    y = 0.0
    bad = 0.0
    for score, actual in ranked_predictions:
        if actual > 0:
            y += 1
        else:
            bad += y
    AUC = 1.0 - (bad / (num_positive_samples * num_negative_samples))
    return AUC


def calculate_f1_score(
    pred: np.ndarray,
    data0: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.1,
) -> float:
    """
    Calculate the F1 score for the given predictions and data.

    Parameters
    ----------
    pred : np.ndarray
        The predicted adjacency matrix.
    data0 : np.ndarray
        The original adjacency matrix.
    mask : Optional[np.ndarray], optional
        The mask to apply on the data, by default None.
    threshold : float, optional
        The threshold to binarize the predictions, by default 0.1.

    Returns
    -------
    float
        The F1 score for the given predictions and data.
    """
    Z_pred = np.copy(pred[0])
    Z_pred[Z_pred < threshold] = 0
    Z_pred[Z_pred >= threshold] = 1

    data = (data0 > 0).astype("int")
    if mask is None:
        return metrics.f1_score(data.flatten(), Z_pred.flatten())
    else:
        return metrics.f1_score(data[mask], Z_pred[mask])
