"""
It provides essential functions for model assessment like AUC for link prediction, conditional
and marginal expectations.
"""

from typing import Optional, Union

import numpy as np
from scipy.stats import poisson
from sklearn import metrics
from sktensor import dtensor, sptensor

from ..input.tools import check_symmetric, transpose_ij, transpose_ij2, transpose_ij3


def calculate_AUC(
    pred: np.ndarray, data0: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """
    Return the AUC of the link prediction. It represents the probability that a randomly chosen
    missing connection (true positive) is given a higher score by our method than a randomly chosen
    pair of unconnected vertices (true negative).

    Parameters
    ----------
    pred : ndarray
           Inferred values.
    data0 : ndarray
            Given values.
    mask : ndarray
           Mask for selecting a subset of the adjacency tensor.

    Returns
    -------
    AUC value.
    """
    # The following line is needed to avoid a bug in sklearn
    data = (data0 > 0).astype("int")

    if mask is None:
        fpr, tpr, _ = metrics.roc_curve(data.flatten(), pred.flatten())
    else:
        fpr, tpr, _ = metrics.roc_curve(data[mask > 0], pred[mask > 0])

    return metrics.auc(fpr, tpr)

def calculate_AUC_mtcov(B, u, v, w, mask=None):
    """
        Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
        (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
        (true negative).

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        u : ndarray
            Membership matrix (out-degree).
        v : ndarray
            Membership matrix (in-degree).
        w : ndarray
            Affinity tensor.
        mask : ndarray
               Mask for selecting a subset of the adjacency tensor.

        Returns
        -------
        AUC : float
              AUC value.
    """

    M = expected_Aija_mtcov(u, v, w)

    if mask is None:
        R = list(zip(M.flatten(), B.flatten()))
        Pos = B.sum()
    else:
        R = list(zip(M[mask > 0], B[mask > 0]))
        Pos = B[mask > 0].sum()

    R.sort(key=lambda x: x[0], reverse=False)
    R_length = len(R)
    Neg = R_length - Pos

    return fAUC(R, Pos, Neg)

def fAUC(R, Pos, Neg):
    y = 0.
    bad = 0.
    for m, a in R:
        if (a > 0):
            y += 1
        else:
            bad += y

    AUC = 1. - (bad / (Pos * Neg))
    return AUC


def calculate_AUC_mtcov(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
    (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
    (true negative).

    Parameters
    ----------
    B : ndarray
        Graph adjacency tensor.
    u : ndarray
        Membership matrix (out-degree).
    v : ndarray
        Membership matrix (in-degree).
    w : ndarray
        Affinity tensor.
    mask : ndarray
           Mask for selecting a subset of the adjacency tensor.

    Returns
    -------
    AUC : float
          AUC value.
    """

    M = expected_Aija_mtcov(u, v, w)

    if mask is None:
        R = list(zip(M.flatten(), B.flatten()))
        Pos = B.sum()
    else:
        R = list(zip(M[mask > 0], B[mask > 0]))
        Pos = B[mask > 0].sum()

    R.sort(key=lambda x: x[0], reverse=False)
    R_length = len(R)
    Neg = R_length - Pos

    return fAUC(R, Pos, Neg)


def fAUC(R: list, Pos: float, Neg: float) -> float:
    """
    Compute the Area Under the Curve (AUC) for the given ranked list of predictions.

    Parameters
    ----------
    R : list
        List of tuples containing the predicted scores and actual labels.
    Pos : float
        Number of positive samples.
    Neg : float
        Number of negative samples.

    Returns
    -------
    float
        The calculated AUC value.
    """
    y = 0.0
    bad = 0.0
    for m, a in R:
        if a > 0:
            y += 1
        else:
            bad += y

    AUC = 1.0 - (bad / (Pos * Neg))
    return AUC


def calculate_conditional_expectation(
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eta: float,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the conditional expectations, e.g. the parameters of the conditional distribution
    lambda_{ij}.

    Parameters
    ----------
    B : ndarray
        Graph adjacency tensor.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.
    eta : float
          Reciprocity coefficient.
    mean : ndarray
           Matrix with mean entries.

    Returns
    -------
    Matrix whose elements are lambda_{ij}.
    """

    if mean is None:
        return lambda_full(u, v, w) + eta * transpose_ij3(
            B
        )  # conditional expectation (knowing A_ji)

    return lambda_full(u, v, w) + eta * transpose_ij3(mean)


def calculate_conditional_expectation_dyncrep(
    B_to_T: Union[dtensor, sptensor],
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eta: float = 0.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Compute the conditional expectations, e.g. the parameters of the conditional  distribution
    lambda_{ij}.

    Parameters
    ----------
    B_to_T : ndarray
        Graph adjacency tensor.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.
    eta : float
          Reciprocity coefficient.
    beta : float
          rate of edge removal.

    Returns
    -------
    Matrix whose elements are lambda_{ij}.
    """
    conditional_expectation = _lambda0_full_dyncrep(u, v, w) + eta * transpose_ij2(
        B_to_T
    )
    M = (beta * conditional_expectation) / (1.0 + beta * conditional_expectation)
    return M


def calculate_expectation(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, eta: float
) -> np.ndarray:
    """
    Compute the expectations, e.g. the parameters of the marginal distribution m_{ij}.

    Parameters
    ----------
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.
    eta : float
          Reciprocity coefficient.

    Returns
    -------
    M : ndarray
        Matrix whose elements are m_{ij}.
    """

    lambda0 = lambda_full(u, v, w)
    lambda0T = transpose_ij3(lambda0)
    M = (lambda0 + eta * lambda0T) / (1.0 - eta * eta)

    return M


def lambda_full(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the mean lambda for all entries (former Exp_ija_matrix(u, v, w)).

    Parameters
    ----------
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.

    Returns
    -------
    M : ndarray
        Mean lambda0 for all entries.
    """

    if w.ndim == 2:
        M = np.einsum("ik,jk->ijk", u, v)
        M = np.einsum("ijk,ak->aij", M, w)
    else:
        M = np.einsum("ik,jq->ijkq", u, v)
        M = np.einsum("ijkq,akq->aij", M, w)

    return M


def _lambda0_full_dyncrep(u, v, w):
    """
        Compute the mean lambda0 for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        M : ndarray
            Mean lambda0 for all entries.
    """

    if w.ndim == 2:
        M = np.einsum("ik,jk->ijk", u, v)
        M = np.einsum("ijk,ak->ij", M, w)
    else:
        M = np.einsum("ik,jq->ijkq", u, v)
        M = np.einsum("ijkq,akq->ij", M, w)

    return M


def calculate_Z(lambda0_aij: np.ndarray, eta: float) -> np.ndarray:
    """
    Compute the normalization constant of the Bivariate Bernoulli distribution.

    Returns
    -------
    Z : ndarray
        Normalization constant Z of the Bivariate Bernoulli distribution.
    """

    Z = (
        lambda0_aij
        + transpose_ij3(lambda0_aij)
        + eta * np.einsum("aij,aji->aij", lambda0_aij, lambda0_aij)
        + 1
    )
    for _, z in enumerate(Z):
        assert check_symmetric(z)

    return Z


def expected_Aija(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the expected value of the adjacency tensor.

    Parameters
    ----------
    U : np.ndarray
        Out-going membership matrix.
    V : np.ndarray
        In-coming membership matrix.
    W : np.ndarray
        Affinity tensor.

    Returns
    -------
    np.ndarray
        The expected value of the adjacency tensor.
    """
    if W.ndim == 1:
        M = np.einsum("ik,jk->ijk", U, V)
        M = np.einsum("ijk,k->ij", M, W)
    else:
        M = np.einsum("ik,jq->ijkq", U, V)
        M = np.einsum("ijkq,kq->ij", M, W)
    return M


def expected_Aija_mtcov(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the expected value of the adjacency tensor for multi-covariate data.

    Parameters
    ----------
    u : np.ndarray
        Out-going membership matrix.
    v : np.ndarray
        In-coming membership matrix.
    w : np.ndarray
        Affinity tensor.

    Returns
    -------
    np.ndarray
        The expected value of the adjacency tensor.
    """
    M = np.einsum("ik,jq->ijkq", u, v)
    M = np.einsum("ijkq,akq->aij", M, w)
    return M

def expected_Aija_mtcov(u, v, w):
    M = np.einsum('ik,jq->ijkq', u, v)
    M = np.einsum('ijkq,akq->aij', M, w)
    return M

def compute_M_joint(U: np.ndarray, V: np.ndarray, W: np.ndarray, eta: float) -> list:
    """
    Return the vectors of joint probabilities of every pair of edges.

    Parameters
    ----------
    U : ndarray
        Out-going membership matrix.
    V : ndarray
        In-coming membership matrix.
    W : ndarray
        Affinity tensor.
    eta : float
          Pair interaction coefficient.

    Returns
    -------
    [p00, p01, p10, p11] : list
                           List of ndarray with joint probabilities of having no edges, only one
                           edge in one direction and both edges for every pair of edges.
    """

    lambda0_aij = lambda_full(U, V, W)

    Z = calculate_Z(lambda0_aij, eta)

    p00 = 1 / Z
    p10 = lambda0_aij / Z
    p01 = transpose_ij3(p10)
    p11 = (eta * lambda0_aij * transpose_ij3(lambda0_aij)) / Z

    return [p00, p01, p10, p11]


def func_lagrange_multiplier(lambda_i: float, num: float, den: float) -> float:
    """
    Function to calculate the value of the Lagrange multiplier.

    Parameters
    ----------
    lambda_i : float
        The current value of the Lagrange multiplier.
    num : float
        The numerator of the function.
    den : float
        The denominator of the function.

    Returns
    -------
    float
        The calculated value of the function.
    """
    f = num / (lambda_i + den)
    return np.sum(f) - 1


def u_with_lagrange_multiplier(
    u: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Function to update the membership matrix 'u' using the Lagrange multiplier.

    Parameters
    ----------
    u : ndarray
        The current membership matrix 'u'.
    x : ndarray
        The first operand in the calculation.
    y : ndarray
        The second operand in the calculation.

    Returns
    -------
    ndarray
        The updated membership matrix 'u'.
    """
    denominator = x.sum() - (y * u).sum()
    f_ui = x / (y + denominator)
    if (u < 0).sum() > 0:
        return 100.0 * np.ones(u.shape)
    return f_ui - u


def expected_computation(
    B: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray, eta: float
) -> tuple:
    """
    Return the marginal and conditional expected value.

    Parameters
    ----------
    B : ndarray
        Graph adjacency tensor.
    U : ndarray
        Out-going membership matrix.
    V : ndarray
        In-coming membership matrix.
    W : ndarray
        Affinity tensor.
    eta : float
          Pair interaction coefficient.

    Returns
    -------
    M_marginal : ndarray
                 Marginal expected values.
    M_conditional : ndarray
                    Conditional expected values.
    """

    lambda0_aij = lambda_full(U, V, W)
    L = lambda0_aij.shape[0]

    Z = calculate_Z(lambda0_aij, eta)
    M_marginal = (lambda0_aij + eta * lambda0_aij * transpose_ij3(lambda0_aij)) / Z
    for layer in np.arange(L):
        np.fill_diagonal(M_marginal[layer], 0.0)

    M_conditional = (eta ** transpose_ij3(B) * lambda0_aij) / (
        eta ** transpose_ij3(B) * lambda0_aij + 1
    )
    for layer in np.arange(L):
        np.fill_diagonal(M_conditional[layer], 0.0)

    return M_marginal, M_conditional

def CalculatePermutation(U_infer: np.ndarray, U0: np.ndarray) -> np.ndarray:
    """
    Permute the overlap matrix so that the groups from the two partitions correspond.

    Parameters
    ----------
    U_infer : np.ndarray
        Inferred membership matrix.
    U0 : np.ndarray
        Reference membership matrix with dimensions NxK.

    Returns
    -------
    np.ndarray
        Permutation matrix.
    """
    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)  # dim=RANKxRANK
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # Permutation matrix
    for _ in range(RANK):
        # Find the max element in the remaining submatrix,
        # the one with rows and columns removed from previous iterations
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

    Parameters
    ----------
    U_infer : np.ndarray
        Inferred membership matrix.
    U0 : np.ndarray
        Reference membership matrix with dimensions NxK.

    Returns
    -------
    tuple
        Permuted inferred matrix and cosine similarity value.
    """
    P = CalculatePermutation(U_infer, U0)
    U_infer = np.dot(U_infer, P)  # Permute inferred matrix
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
    AT = transpose_ij(A)
    MT = transpose_ij(M)
    num = (mu + EPS) * poisson.pmf(A, (pi + EPS)) * poisson.pmf(AT, (pi + EPS))
    # num = poisson.pmf(A,pi) * poisson.pmf(AT,pi)* (mu+EPS)
    den = num + poisson.pmf(A, M) * poisson.pmf(AT, MT) * (1 - mu + EPS)
    if mask is None:
        return num / den
    else:
        return num[mask.nonzero()] / den[mask.nonzero()]


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
        Predicted values.
    data0 : np.ndarray
        True values.
    mask : Optional[np.ndarray]
        Mask for selecting a subset of the data.
    threshold : float
        Threshold for binarizing the predictions.

    Returns
    -------
    float
        The F1 score.
    """
    Z_pred = np.copy(pred[0])
    Z_pred[Z_pred < threshold] = 0
    Z_pred[Z_pred >= threshold] = 1

    data = (data0 > 0).astype("int")
    if mask is None:
        return metrics.f1_score(data.flatten(), Z_pred.flatten())
    else:
        return metrics.f1_score(data[mask], Z_pred[mask])


def calculate_expectation_acd(
    U: np.ndarray, V: np.ndarray, W: np.ndarray, Q: np.ndarray, pi: float = 1
) -> np.ndarray:
    """
    Calculate the expectation for the adjacency tensor with an additional covariate.

    Parameters
    ----------
    U : np.ndarray
        Out-going membership matrix.
    V : np.ndarray
        In-coming membership matrix.
    W : np.ndarray
        Affinity tensor.
    Q : np.ndarray
        Covariate matrix.
    pi : float, optional
        Poisson parameter, by default 1.

    Returns
    -------
    np.ndarray
        The calculated expectation.
    """
    lambda0 = lambda_full(U, V, W)
    return (1 - Q) * lambda0 + Q * pi
