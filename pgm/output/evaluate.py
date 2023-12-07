"""
It provides essential functions for model assessment like AUC for link prediction, conditional
and marginal expectations and the pseudo log-likelihood of the data.
"""
import numpy as np
from sklearn import metrics

from ..input.tools import transpose_ij3

# TODO: make it model agnostic


def calculate_AUC(pred, data0, mask=None):
    """
        Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
        (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
        (true negative).

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

    data = (data0 > 0).astype('int')
    if mask is None:
        fpr, tpr, thresholds = metrics.roc_curve(data.flatten(),
                                                 pred.flatten())
    else:
        fpr, tpr, thresholds = metrics.roc_curve(data[mask > 0],
                                                 pred[mask > 0])

    return metrics.auc(fpr, tpr)


def calculate_conditional_expectation(B, u, v, w, eta=0.0, mean=None):
    """
        Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

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
        return _lambda0_full(u, v, w) + eta * transpose_ij3(
            B)  # conditional expectation (knowing A_ji)
    else:
        return _lambda0_full(u, v, w) + eta * transpose_ij3(mean)


def calculate_expectation(u, v, w, eta=0.0):
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

    lambda0 = _lambda0_full(u, v, w)
    lambda0T = transpose_ij3(lambda0)
    M = (lambda0 + eta * lambda0T) / (1. - eta * eta)

    return M


def _lambda0_full(u, v, w):  # same as Exp_ija_matrix(u, v, w)
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
        M = np.einsum('ik,jk->ijk', u, v)
        M = np.einsum('ijk,ak->aij', M, w)
    else:
        M = np.einsum('ik,jq->ijkq', u, v)
        M = np.einsum('ijkq,akq->aij', M, w)

    return M


def PSloglikelihood(B, u, v, w, eta, mask=None):
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
        M = _lambda0_full(u, v, w)
        M += (eta * B[0, :, :].T)[np.newaxis, :, :]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()
    else:
        M = _lambda0_full(u, v, w)[mask > 0]
        M += (eta * B[0, :, :].T)[np.newaxis, :, :][mask > 0]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B[mask > 0] * logM).sum() - M.sum()


# TODO: make it model agnostic


def calculate_opt_func(B, algo_obj=None, mask=None, assortative=False):
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
        B_test[np.logical_not(mask)] = 0.

    if not assortative:
        return PSloglikelihood(B,
                               algo_obj.u_f,
                               algo_obj.v_f,
                               algo_obj.w_f,
                               algo_obj.eta_f,
                               mask=mask)
    else:
        L = B.shape[0]
        K = algo_obj.w_f.shape[-1]
        w = np.zeros((L, K, K))
        for l in range(L):
            w1 = np.zeros((K, K))
            np.fill_diagonal(w1, algo_obj.w_f[l])
            w[l, :, :] = w1.copy()
        return PSloglikelihood(B,
                               algo_obj.u_f,
                               algo_obj.v_f,
                               w,
                               algo_obj.eta_f,
                               mask=mask)



def calculate_Z(lambda0_aij, eta):
    """
        Compute the normalization constant of the Bivariate Bernoulli distribution.

        Returns
        -------
        Z : ndarray
            Normalization constant Z of the Bivariate Bernoulli distribution.
    """

    Z = lambda0_aij + transpose_ij3(lambda0_aij) + eta * np.einsum('aij,aji->aij', lambda0_aij, lambda0_aij) + 1
    for l in range(len(Z)):
        assert check_symmetric(Z[l])

    return Z


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
        Check if a matrix a is symmetric.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def compute_M_joint(U, V, W, eta):
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
                               List of ndarray with joint probabilities of having no edges, only one edge in one
                               direction and both edges for every pair of edges.
    """

    lambda0_aij = _lambda0_full(U, V, W)

    Z = calculate_Z(lambda0_aij, eta)

    p00 = 1 / Z
    p10 = lambda0_aij / Z
    p01 = transpose_ij3(p10)
    p11 = (eta * lambda0_aij * transpose_ij3(lambda0_aij)) / Z

    return [p00, p01, p10, p11]

def expected_computation(B, U, V, W, eta):
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

    lambda0_aij = _lambda0_full(U, V, W)
    L = lambda0_aij.shape[0]

    Z = calculate_Z(lambda0_aij, eta)
    M_marginal = (lambda0_aij + eta * lambda0_aij * transpose_ij3(lambda0_aij)) / Z
    for l in np.arange(L):
        np.fill_diagonal(M_marginal[l], 0.)

    M_conditional = (eta ** transpose_ij3(B) * lambda0_aij) / (eta ** transpose_ij3(B) * lambda0_aij + 1)
    for l in np.arange(L):
        np.fill_diagonal(M_conditional[l], 0.)

    return M_marginal, M_conditional
