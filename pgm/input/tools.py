"""
It provides utility functions to handle matrices, tensors, and sparsity. It includes functions for
checking if an object can be cast to an integer, normalizing matrices, determining sparsity in
tensors, and converting between dense and sparse representations.
"""
import numpy as np
import sktensor as skt


def can_cast(string):
    """
        Verify if one object can be converted to integer object.

        Parameters
        ----------
        string : int or float or str
                 Name of the node.

        Returns
        -------
        bool : bool
               If True, the input can be converted to integer object.
    """

    try:
        int(string)
        return True
    except ValueError:
        return False


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.

        Parameters
        ----------
        u: ndarray
           Numpy Matrix.

        Returns
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def get_item_array_from_subs(A, ref_subs):
    """
        Get values of ref_subs entries of a dense tensor.
        Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


# TODO: merge into one
def transpose_ij3(M):
    """
        Compute the transpose of a tensor.

        Parameters
        ----------
        M : ndarray
            Numpy array.

        Returns
        -------
        Transpose of the tensor.
    """

    return np.einsum('aij->aji', M)


def transpose_ij2(M):
    """
        Compute the transpose of a matrix.

        Parameters
        ----------
        M : ndarray
            Numpy matrix.

        Returns
        -------
        Transpose of the matrix.
    """

    return np.einsum('ij->ji', M)


def Exp_ija_matrix(u, v, w):
    """
        Compute the mean lambda0_ij for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity matrix.

        Returns
        -------
        M : ndarray
            Mean lambda0_ij for all entries.
    """

    M = np.einsum('ik,jq->ijkq', u, v)
    M = np.einsum('ijkq,kq->ij', M, w)

    return M
