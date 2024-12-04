"""
Functions for evaluating link prediction.
"""

from functools import singledispatch
from typing import Optional, Union

import numpy as np
from sklearn import metrics

from probinet.evaluation.expectation_computation import (
    compute_expected_adjacency_tensor_multilayer,
)


@singledispatch
def mask_or_flatten_array(
    mask: Union[np.ndarray, None], expected_adjacency: np.ndarray
) -> np.ndarray:
    raise NotImplementedError(f"Unsupported type {type(mask)} for mask.")


@mask_or_flatten_array.register(type(None))
def _(mask: None, expected_adjacency: np.ndarray) -> np.ndarray:
    return expected_adjacency.flatten()


@mask_or_flatten_array.register(np.ndarray)
def _(mask: np.ndarray, expected_adjacency: np.ndarray) -> np.ndarray:
    return expected_adjacency[mask > 0]


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
    processed_data = mask_or_flatten_array(mask, data)
    processed_pred = mask_or_flatten_array(mask, pred)
    fpr, tpr, _ = metrics.roc_curve(processed_data, processed_pred)
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
    # Flatten the expected adjacency tensor and the predicted adjacency tensor
    processed_adjacency = mask_or_flatten_array(
        mask,
        expected_adjacency,
    )
    processed_B = mask_or_flatten_array(mask, B)
    # Combine the processed adjacency tensor and the processed predicted adjacency tensor
    ranked_predictions = list(zip(processed_adjacency, processed_B))
    # Calculate the number of positive samples
    num_positive_samples = processed_B.sum()
    # Sort the ranked predictions in ascending order
    ranked_predictions.sort(key=lambda x: x[0], reverse=False)
    # Calculate the AUC value
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
    for _, actual in ranked_predictions:
        if actual > 0:
            y += 1
        else:
            bad += y
    AUC = 1.0 - (bad / (num_positive_samples * num_negative_samples))
    return AUC


def calculate_f1_score(
    data0: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.1,
) -> float:
    """
    Calculate the F1 score for the given predictions and data.

    Parameters
    ----------
    data0 : np.ndarray
        The original adjacency matrix.
    pred : np.ndarray
        The predicted adjacency matrix.
    mask : Optional[np.ndarray], optional
        The mask to apply on the data, by default None.
    threshold : float, optional
        The threshold to binarize the predictions, by default 0.1.

    Returns
    -------
    float
        The F1 score for the given predictions and data.
    """
    # Binarize the predictions based on the threshold
    Z_pred = np.copy(pred[0])
    Z_pred[Z_pred < threshold] = 0
    Z_pred[Z_pred >= threshold] = 1

    # Binarize the data
    data = (data0 > 0).astype("int")

    return metrics.f1_score(
        mask_or_flatten_array(mask, data), mask_or_flatten_array(mask, Z_pred)
    )
