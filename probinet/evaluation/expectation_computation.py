"""
Functions for computing expectations and related metrics.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn import metrics

from ..evaluation.covariate_prediction import extract_true_label, predict_label
from ..types import GraphDataType
from ..utils.matrix_operations import transpose_matrix, transpose_tensor
from ..utils.tools import check_symmetric


def calculate_conditional_expectation(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eta: float,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates the conditional expectation of a given set of parameters.

    This function computes the conditional expectation of a multivariate random
    process based on the provided inputs. It incorporates a scaling parameter
    (eta) and optionally a mean tensor to compute the result. If the mean tensor
    is not provided, it defaults to computing a mean based on the input parameters
    u, v, and w.

    Parameters
    ----------
    B
        A 3-dimensional tensor used in the computation of the weighted mean. The
        tensor represents the relationship between variables in the multivariate
        process.
    u
        A 1-dimensional array representing the first set of variables
        contributing to the computation of the mean. Typically corresponds to a
        principal factor in the model.
    v
        A 1-dimensional array representing the second set of variables
        contributing to the computation of the mean. It complements the u array
        in forming the joint distribution.
    w
        A 1-dimensional array representing the third set of variables
        interacting with u and v to establish a comprehensive measure of
        centrality in the model.
    eta
        A scaling parameter applied to the tensor factors to adjust their
        weighted contribution to the overall mean measure of expectations.
    mean
        A precomputed 3-dimensional tensor mean to override the default computed
        mean. If None, the mean will be computed using u, v, and w.

    Returns
    -------
    np.ndarray
        A 3-dimensional tensor that represents the computed conditional
        expectation given the provided parameters. The tensor provides a
        context-sensitive measure of expectation, calculated as either a default
        weighted mean or incorporating a provided precomputed mean.
    """
    if mean is None:
        return compute_mean_lambda0(u, v, w) + eta * transpose_tensor(B)
    return compute_mean_lambda0(u, v, w) + eta * transpose_tensor(mean)


def calculate_conditional_expectation_dyncrep(
    B_to_T: GraphDataType,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eta: float = 0.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Calculate the conditional expectation for given dynamic representation.

    This function computes the conditional expectation based on the dynamic
    representation of the input data, including arrays and graph-based data.
    It utilizes transformation and normalization techniques such as
    matrix transposition and scaling.

    Parameters
    ----------
    B_to_T : GraphDataType
        Graph-based data structure representing relationships or
        transitions between nodes or states.

    u : np.ndarray
        Input array representing data points or state variables.

    v : np.ndarray
        Input array representing another set of data points or state
        variables related to `u`.

    w : np.ndarray
        Auxiliary input array used for reference in the calculation.

    eta : float, optional
        Scaling factor applied to the graph transformation. Default is 0.0.

    beta : float, optional
        Scaling factor applied in the normalization computation. Default
        is 1.0.

    Returns
    -------
    np.ndarray
        Array representing the computed conditional expectation
        after applying the dynamic representation transformations.
    """
    conditional_expectation = compute_mean_lambda0_dyncrep(
        u, v, w
    ) + eta * transpose_matrix(B_to_T)
    M = (beta * conditional_expectation) / (1.0 + beta * conditional_expectation)
    return M


def calculate_expectation(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, eta: float
) -> np.ndarray:
    """
    Compute the expectations for the marginal distribution.
    """
    lambda0 = compute_mean_lambda0(u, v, w)
    lambda0T = transpose_tensor(lambda0)
    M = (lambda0 + eta * lambda0T) / (1.0 - eta * eta)
    return M


def compute_mean_lambda0(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the mean lambda0 for all entries.
    """
    if w.ndim == 2:
        M = np.einsum("ik,jk->ijk", u, v)
        M = np.einsum("ijk,ak->aij", M, w)
    else:
        M = np.einsum("ik,jq->ijkq", u, v)
        M = np.einsum("ijkq,akq->aij", M, w)
    return M


def compute_mean_lambda0_dyncrep(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Compute the mean lambda0 for all entries for dynamic reciprocity.
    """
    if w.ndim == 2:
        M = np.einsum("ik,jk->ijk", u, v)
        M = np.einsum("ijk,ak->ij", M, w)
    else:
        M = np.einsum("ik,jq->ijkq", u, v)
        M = np.einsum("ijkq,akq->ij", M, w)
    return M


def compute_mean_lambda0_nonzero(
    subs_nz: Tuple[int, int, int],
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    assortative=True,
) -> np.ndarray:
    """
    Compute the mean lambda0_ij for only non-zero entries.
    """
    if not assortative:
        nz_recon_IQ = np.einsum("Ik,Ikq->Iq", u[subs_nz[1], :], w[subs_nz[0], :, :])
    else:
        nz_recon_IQ = np.einsum("Ik,Ik->Ik", u[subs_nz[1], :], w[subs_nz[0], :])
    nz_recon_I = np.einsum("Iq,Iq->I", nz_recon_IQ, v[subs_nz[2], :])
    return nz_recon_I


def calculate_Z(lambda0_aij: np.ndarray, eta: float) -> np.ndarray:
    """
    Compute the normalization constant of the Bivariate Bernoulli distribution.
    """
    Z = (
        lambda0_aij
        + transpose_tensor(lambda0_aij)
        + eta * np.einsum("aij,aji->aij", lambda0_aij, lambda0_aij)
        + 1
    )
    for _, z in enumerate(Z):
        assert check_symmetric(z)
    return Z


def compute_expected_adjacency_tensor(
    U: np.ndarray, V: np.ndarray, W: np.ndarray
) -> np.ndarray:
    """
    Compute the expected value of the adjacency tensor.
    """
    if W.ndim == 1:
        M = np.einsum("ik,jk->ijk", U, V)
        M = np.einsum("ijk,k->ij", M, W)
    else:
        M = np.einsum("ik,jq->ijkq", U, V)
        M = np.einsum("ijkq,kq->ij", M, W)
    return M


def compute_expected_adjacency_tensor_multilayer(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Compute the expected value of the adjacency tensor for multi-covariate data.
    """
    M = np.einsum("ik,jq->ijkq", u, v)
    M = np.einsum("ijkq,akq->aij", M, w)
    return M


def compute_M_joint(U: np.ndarray, V: np.ndarray, W: np.ndarray, eta: float) -> list:
    """
    Return the vectors of joint probabilities of every pair of edges.
    """
    # Compute the mean lambda0 for all entries
    lambda0_aij = compute_mean_lambda0(U, V, W)

    # Calculate the normalization constant Z
    Z = calculate_Z(lambda0_aij, eta)

    # Compute the joint probability p00 (both edges are absent)
    p00 = 1 / Z

    # Compute the joint probability p10 (first edge is present, second is absent)
    p10 = lambda0_aij / Z

    # Compute the joint probability p01 (first edge is absent, second is present)
    p01 = transpose_tensor(p10)

    # Compute the joint probability p11 (both edges are present)
    p11 = (eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z

    # Return the list of joint probabilities
    return [p00, p01, p10, p11]


def compute_lagrange_multiplier(lambda_i: float, num: float, den: float) -> float:
    """
    Function to calculate the value of the Lagrange multiplier.
    """
    f = num / (lambda_i + den)
    return np.sum(f) - 1


def u_with_lagrange_multiplier(
    u: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Function to update the membership matrix 'u' using the Lagrange multiplier.
    """
    denominator = x.sum() - (y * u).sum()
    f_ui = x / (y + denominator)
    if (u < 0).sum() > 0:
        return 100.0 * np.ones(u.shape)
    return f_ui - u


def compute_marginal_and_conditional_expectation(
    B: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray, eta: float
) -> tuple:
    """
    Return the marginal and conditional expected value.
    """
    lambda0_aij = compute_mean_lambda0(U, V, W)
    L = lambda0_aij.shape[0]
    Z = calculate_Z(lambda0_aij, eta)
    M_marginal = (lambda0_aij + eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z
    for layer in np.arange(L):
        np.fill_diagonal(M_marginal[layer], 0.0)
    M_conditional = (eta ** transpose_tensor(B) * lambda0_aij) / (
        eta ** transpose_tensor(B) * lambda0_aij + 1
    )
    for layer in np.arange(L):
        np.fill_diagonal(M_conditional[layer], 0.0)
    return M_marginal, M_conditional


def calculate_expectation_acd(
    U: np.ndarray, V: np.ndarray, W: np.ndarray, Q: np.ndarray, pi: float = 1
) -> np.ndarray:
    """
    Calculate the expectation for the adjacency tensor with an additional covariate.
    """
    lambda0 = compute_mean_lambda0(U, V, W)
    return (1 - Q) * lambda0 + Q * pi


def calculate_Q_dense(
    A: np.ndarray,
    M: np.ndarray,
    pi: float,
    mu: float,
    mask: Optional[np.ndarray] = None,
    EPS: float = 1e-12,
) -> np.ndarray:
    """
    Compute the dense Q matrix for the given adjacency tensor and parameters.

    Parameters
    ----------
    A : np.ndarray
        Adjacency tensor.
    M : np.ndarray
        Mean adjacency tensor.
    pi : float
        Poisson parameter.
    mu : float
        Mixing parameter.
    mask : Optional[np.ndarray]
        Mask for selecting a subset of the adjacency tensor.
    EPS : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Dense Q matrix.
    """
    AT = transpose_tensor(A)
    MT = transpose_tensor(M)
    num = (mu + EPS) * poisson.pmf(A, (pi + EPS)) * poisson.pmf(AT, (pi + EPS))
    # num = poisson.pmf(A,pi) * poisson.pmf(AT,pi)* (mu+EPS)
    den = num + poisson.pmf(A, M) * poisson.pmf(AT, MT) * (1 - mu + EPS)
    if mask is None:
        return num / den
    else:
        return num[mask.nonzero()] / den[mask.nonzero()]


def compute_L1loss(X: np.ndarray, Xtilde: np.ndarray) -> float:
    """
    Calculate the L1 loss between two matrices.

    Parameters
    ----------
    X : np.ndarray
        The first matrix.
    Xtilde : np.ndarray
        The second matrix to compare against the first matrix.

    Returns
    -------
    float
        The L1 loss between the two matrices, rounded to three decimal places.
    """
    return np.round(np.mean(np.abs(X - Xtilde)), 3)
