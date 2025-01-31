"""
This module provides functions for computing the log-likelihood and pseudo log-likelihood of the models, as well as other related calculations.
"""

from typing import Optional

import numpy as np
import pandas as pd

from probinet.evaluation.expectation_computation import (
    compute_expected_adjacency_tensor,
    compute_expected_adjacency_tensor_multilayer,
    compute_mean_lambda0,
)
from probinet.utils.tools import log_and_raise_error


def loglikelihood(
    B: np.ndarray,
    X: pd.DataFrame,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    gamma: float,
    maskG: Optional[np.ndarray] = None,
    maskX: Optional[np.ndarray] = None,
) -> float:
    """
    Return the log-likelihood of the model.

    Parameters
    ----------
    B : np.ndarray
        Graph adjacency tensor.
    X : pd.DataFrame
        Pandas DataFrame object representing the one-hot encoding version of the design matrix.
    u : np.ndarray
        Membership matrix (out-degree).
    v : np.ndarray
        Membership matrix (in-degree).
    w : np.ndarray
        Affinity tensor.
    beta : np.ndarray
        Beta parameter matrix.
    gamma : float
        Scaling parameter gamma.
    maskG : Optional[np.ndarray]
        Mask for selecting a subset in the adjacency tensor.
    maskX : Optional[np.ndarray]
        Mask for selecting a subset in the design matrix.

    Returns
    -------
    float
        Log-likelihood value.
    """
    # Compute the log-likelihood for the attributes
    attr = loglikelihood_attributes(X, u, v, beta, mask=maskX)

    # Compute the log-likelihood for the network structure
    graph = loglikelihood_network(B, u, v, w, mask=maskG)

    # Combine the two log-likelihoods using the scaling parameter gamma
    loglik = (1 - gamma) * graph + gamma * attr

    return loglik


def loglikelihood_network(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the log-likelihood for the network structure.

    Parameters
    ----------
    B : np.ndarray
        Graph adjacency tensor.
    u : np.ndarray
        Membership matrix (out-degree).
    v : np.ndarray
        Membership matrix (in-degree).
    w : np.ndarray
        Affinity tensor.
    mask : Optional[np.ndarray]
        Mask for selecting a subset in the adjacency tensor.

    Returns
    -------
    float
        Log-likelihood value for the network structure.
    """
    if mask is None:
        # Compute the expected adjacency tensor
        M = compute_expected_adjacency_tensor(u, v, w)
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()

    # Compute the expected adjacency tensor for the masked elements
    M = compute_expected_adjacency_tensor_multilayer(u, v, w)[mask > 0]
    logM = np.zeros(M.shape)
    logM[M > 0] = np.log(M[M > 0])
    return (B[mask > 0] * logM).sum() - M.sum()


def loglikelihood_attributes(
    X: pd.DataFrame,
    u: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the log-likelihood for the attributes.

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
        Mask for selecting a subset in the design matrix.

    Returns
    -------
    float
        Log-likelihood value for the attributes.
    """
    if mask is None:
        # Compute the expected attribute matrix
        p = np.dot(u + v, beta) / 2
        nonzeros = p > 0.0
        p[nonzeros] = np.log(p[nonzeros] / 2.0)
        return (X * p).sum().sum()

    # Compute the expected attribute matrix for the masked elements
    p = np.dot(u[mask > 0] + v[mask > 0], beta) / 2
    nonzeros = p > 0.0
    p[nonzeros] = np.log(p[nonzeros] / 2.0)
    return (X.iloc[mask > 0] * p).sum().sum()


def likelihood_conditional(
    M: np.ndarray,
    beta: float,
    data: np.ndarray,
    data_tm1: np.ndarray,
    EPS: Optional[float] = 1e-10,
) -> float:
    """
    Compute the log-likelihood of the data given the parameters.

    Parameters
    ----------
    M : np.ndarray
        Matrix of expected values.
    beta : float
        Rate of edge removal.
    data : np.ndarray
        Current adjacency tensor.
    data_tm1 : np.ndarray
        Previous adjacency tensor.
    EPS : float, optional
        Small constant to prevent division by zero and log of zero.

    Returns
    -------
    float
        Log-likelihood value.
    """
    # Initialize the log-likelihood
    l = -M.sum()

    # Compute the log-likelihood for the non-zero elements
    sub_nz_and = np.logical_and(data > 0, (1 - data_tm1) > 0)
    Alog = data[sub_nz_and] * (1 - data_tm1)[sub_nz_and] * np.log(M[sub_nz_and] + EPS)
    l += Alog.sum()

    # Compute the log-likelihood for the elements that are present in both data and data_tm1
    sub_nz_and = np.logical_and(data > 0, data_tm1 > 0)
    l += np.log(1 - beta + EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()

    # Compute the log-likelihood for the elements that are present in data_tm1 but not in data
    sub_nz_and = np.logical_and(data_tm1 > 0, (1 - data) > 0)
    l += np.log(beta + EPS) * ((1 - data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()

    # Check for NaN values in the log-likelihood
    if np.isnan(l):
        log_and_raise_error(ValueError, "Likelihood is NaN!")
    return l


def PSloglikelihood(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eta: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the pseudo log-likelihood of the data.

    Parameters
    ----------
    B : np.ndarray
        Graph adjacency tensor.
    u : np.ndarray
        Out-going membership matrix.
    v : np.ndarray
        In-coming membership matrix.
    w : np.ndarray
        Affinity tensor.
    eta : float
        Reciprocity coefficient.
    mask : Optional[np.ndarray]
        Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

    Returns
    -------
    float
        Pseudo log-likelihood value.
    """
    if mask is None:
        # Compute the expected adjacency tensor
        M = compute_mean_lambda0(u, v, w)
        M += (eta * B[0, :, :].T)[np.newaxis, :, :]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()

    # Compute the expected adjacency tensor for the masked elements
    M = compute_mean_lambda0(u, v, w)[mask > 0]
    M += (eta * B[0, :, :].T)[np.newaxis, :, :][mask > 0]
    logM = np.zeros(M.shape)
    logM[M > 0] = np.log(M[M > 0])
    return (B[mask > 0] * logM).sum() - M.sum()


def calculate_opt_func(
    B: np.ndarray,
    algo_obj,
    mask: Optional[np.ndarray] = None,
    assortative: bool = False,
) -> float:
    """
    Compute the optimal value for the pseudo log-likelihood with the inferred parameters.

    Parameters
    ----------
    B : np.ndarray
        Graph adjacency tensor.
    algo_obj : object
        The CRep object.
    mask : Optional[np.ndarray]
        Mask for selecting a subset of the adjacency tensor.
    assortative : bool
        Flag to use an assortative mode.

    Returns
    -------
    float
        Maximum pseudo log-likelihood value.
    """
    # Copy the adjacency tensor
    B_test = B.copy()
    if mask is not None:
        B_test[np.logical_not(mask)] = 0.0

    if not assortative:
        return PSloglikelihood(
            B, algo_obj.u_f, algo_obj.v_f, algo_obj.w_f, algo_obj.eta_f, mask=mask
        )

    # Compute the pseudo log-likelihood in assortative mode
    L = B.shape[0]
    K = algo_obj.w_f.shape[-1]
    w = np.zeros((L, K, K))
    for l in range(L):
        w1 = np.zeros((K, K))
        np.fill_diagonal(w1, algo_obj.w_f[l])
        w[l, :, :] = w1.copy()
    return PSloglikelihood(B, algo_obj.u_f, algo_obj.v_f, w, algo_obj.eta_f, mask=mask)


def log_likelihood_given_model(model: object, adjacency_tensor: np.ndarray) -> float:
    """
    Calculate the log-likelihood of the model considering only the structural data.

    Parameters
    ----------
    model : object
        The model object containing the lambda0_ija and lambda0_nz attributes.
    adjacency_tensor : np.ndarray
        The adjacency matrix.

    Returns
    -------
    float
        The log-likelihood value, rounded to three decimal places.
    """
    M = model.lambda0_ija
    loglikelihood = -M.sum()
    logM = np.log(model.lambda0_nz)
    XlogM = adjacency_tensor[adjacency_tensor.nonzero()] * logM
    loglikelihood += XlogM.sum()
    return np.round(loglikelihood, 3)
