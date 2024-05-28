"""
It provides essential functions for model assessment like AUC for link prediction, conditional
and marginal expectations and the pseudo log-likelihood of the data.
"""

from typing import Optional, Union

import numpy as np
from sklearn import metrics
from sktensor import dtensor, sptensor

from ..input.tools import check_symmetric, transpose_ij2, transpose_ij3
from ..model.constants import EPS_

# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals, too-many-branches,
# too-many-statements
# pylint: disable=fixme


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
    beta : float
          rate of edge removal.
    mean : ndarray
           Matrix with mean entries.

    Returns
    -------
    Matrix whose elements are lambda_{ij}.
    """
    conditional_expectation = lambda_full(u, v, w) + eta * transpose_ij2(B_to_T)
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


# same as Exp_ija_matrix(u, v, w)


def lambda_full(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the mean lambda for all entries.

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
    mask : ndarray
           Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

    Returns
    -------
    Pseudo log-likelihood value.
    """

    if mask is None:
        M = lambda_full(u, v, w)
        M += (eta * B[0, :, :].T)[np.newaxis, :, :]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()

    M = lambda_full(u, v, w)[mask > 0]
    M += (eta * B[0, :, :].T)[np.newaxis, :, :][mask > 0]
    logM = np.zeros(M.shape)
    logM[M > 0] = np.log(M[M > 0])
    return (B[mask > 0] * logM).sum() - M.sum()


# TODO: make it model agnostic


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
    B : ndarray
        Graph adjacency tensor.
    algo_obj : obj
               The CRep object.
    mask : ndarray
           Mask for selecting a subset of the adjacency tensor.
    assortative : bool
                  Flag to use an assortative mode.

    Returns
    -------
    Maximum pseudo log-likelihood value
    """

    B_test = B.copy()
    if mask is not None:
        B_test[np.logical_not(mask)] = 0.0

    if not assortative:
        return PSloglikelihood(
            B, algo_obj.u_f, algo_obj.v_f, algo_obj.w_f, algo_obj.eta_f, mask=mask
        )

    L = B.shape[0]
    K = algo_obj.w_f.shape[-1]
    w = np.zeros((L, K, K))
    for l in range(L):
        w1 = np.zeros((K, K))
        np.fill_diagonal(w1, algo_obj.w_f[l])
        w[l, :, :] = w1.copy()
    return PSloglikelihood(B, algo_obj.u_f, algo_obj.v_f, w, algo_obj.eta_f, mask=mask)


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

def expected_Aija(U, V, W):
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


def Likelihood_conditional(M, beta, data, data_tm1, EPS=EPS_):
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
        l : float
             log-likelihood value.
    """
    l = -M.sum()
    sub_nz_and = np.logical_and(data > 0, (1 - data_tm1) > 0)
    Alog = data[sub_nz_and] * (1 - data_tm1)[sub_nz_and] * np.log(M[sub_nz_and] + EPS)
    l += Alog.sum()
    sub_nz_and = np.logical_and(data > 0, data_tm1 > 0)
    l += np.log(1 - beta + EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    sub_nz_and = np.logical_and(data_tm1 > 0, (1 - data) > 0)
    l += np.log(beta + EPS) * ((1 - data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()
    if np.isnan(l):
        print("Likelihood is NaN!!!!")
        sys.exit(1)
    else:
        return l


def CalculatePermutation(U_infer, U0):
    """
    Permuting the overlap matrix so that the groups from the two partitions correspond
    U0 has dimension NxK, reference memebership
    """
    N, RANK = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N);  # dim=RANKxRANK
    rows = np.zeros(RANK);
    columns = np.zeros(RANK);
    P = np.zeros((RANK, RANK));  # Permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining submatrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.;
        c_index = 1;
        r_index = 1;
        for i in range(RANK):
            if columns[i] == 0:
                for j in range(RANK):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i];
                            c_index = i;
                            r_index = j;

        P[r_index, c_index] = 1;
        columns[c_index] = 1;
        rows[r_index] = 1;

    return P


def cosine_similarity(U_infer, U0):
    """
    It is assumed that matrices are row-normalized
    """
    P = CalculatePermutation(U_infer, U0)
    U_infer = np.dot(U_infer, P)  # Permute infered matrix
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
