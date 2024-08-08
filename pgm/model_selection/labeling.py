from typing import List, Optional

import numpy as np
import pandas as pd


def extract_true_label(
    X: pd.DataFrame, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract true labels from the design matrix X.

    Parameters
    ----------
    X : pd.DataFrame
        Pandas DataFrame object representing the one-hot encoding version of the design matrix.
    mask : Optional[np.ndarray]
        Mask for selecting a subset of the design matrix.

    Returns
    -------
    np.ndarray
        Array of true labels.
    """
    if mask is not None:
        return X.iloc[mask > 0].idxmax(axis=1).values
    else:
        return X.idxmax(axis=1).values


def predict_label(
    X: pd.DataFrame,
    u: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Compute predicted labels.

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
    List[str]
        List of predicted labels.
    """
    if mask is None:
        # Compute the probability of each label
        probs = np.dot((u + v), beta) / 2
        # Check that the sum of the probabilities is equal to the number of examples
        assert np.round(np.sum(np.sum(probs, axis=1)), 0) == u.shape[0]
        # Return the predicted label for each
        return [X.columns[el] for el in np.argmax(probs, axis=1)]
    else:
        # Compute the probability of each label
        probs = np.dot((u[mask > 0] + v[mask > 0]), beta) / 2
        assert np.round(np.sum(np.sum(probs, axis=1)), 0) == u[mask > 0].shape[0]
        # TO REMIND: when gamma=1, this assert fails because we don't update the entries of U and V that belong
        # to the test set, because all these rows will be not in the subs (all elements are zeros)
        return [X.iloc[mask > 0].columns[el] for el in np.argmax(probs, axis=1)]
