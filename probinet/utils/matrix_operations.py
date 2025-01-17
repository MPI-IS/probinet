"""
This module contains functions that perform matrix operations, such as the Khatri-Rao product.
"""

from typing import Optional, Tuple

import numpy as np

from probinet.utils.tools import log_and_raise_error


def sp_uttkrp_assortative(
    vals: np.ndarray,
    subs: Tuple[np.ndarray],
    m: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    temporal: bool = False,
) -> np.ndarray:
    """
    Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

    Parameters
    ----------
    vals : ndarray
           Values of the non-zero entries.
    subs : tuple
           Indices of elements that are non-zero. It is a n-tuple of array-likes and the length
           of tuple n must be equal to the dimension of tensor.
    m : int
        Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
        tensor: if 1 it works with the matrix u; if 2 it works with v.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.
    temporal : bool
        If True, use the static version of the function.

    Returns
    -------
    out : ndarray
          Matrix which is the result of the matrix product of the unfolding of the tensor and
          the Khatri-Rao product of the membership matrix.
    """

    if len(subs) < 3:
        log_and_raise_error(ValueError, "subs_nz should have at least 3 elements.")

    # Determine the shape of the evaluation array
    D, K = u.shape if m == 1 else v.shape
    out = np.zeros((D, K))

    for k in range(K):
        # Copy the values to avoid modifying the original array
        copied_vals = vals.copy()
        # Select the appropriate indices
        indices = subs[0] if temporal else 0
        # Select the appropriate slice of the affinity tensor
        w_I = w[indices, k].astype(copied_vals.dtype)
        # Select the appropriate slice of the membership matrix
        multiplier_u_v = v[subs[2], k] if m == 1 else u[subs[1], k]
        # Transform the multiplier to the same data type as the values
        multiplier_u_v = multiplier_u_v.astype(copied_vals.dtype)
        # Multiply the values by the affinity tensor slice and the membership matrix slice
        copied_vals *= w_I * multiplier_u_v
        # Update the evaluation matrix with the weighted sum of the values
        out[:, k] += np.bincount(subs[m], weights=copied_vals, minlength=D)

    return out


def sp_uttkrp(
    vals: np.ndarray,
    subs: Tuple[np.ndarray],
    m: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    temporal: bool = True,
) -> np.ndarray:
    """
    Compute the Khatri-Rao product (sparse version).

    Parameters
    ----------
    vals : ndarray
           Values of the non-zero entries.
    subs : tuple
           Indices of elements that are non-zero. It is a n-tuple of array-likes and the length
           of tuple n must be equal to the dimension of tensor.
    m : int
        Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
        tensor: if 1 it
        works with the matrix u; if 2 it works with v.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.
    temporal : bool
        Flag to determine if the function should behave in a temporal manner.

    Returns
    -------
    out : ndarray
          Matrix which is the result of the matrix product of the unfolding of the tensor and
          the Khatri-Rao product
          of the membership matrix.
    """

    if len(subs) < 3:
        log_and_raise_error(ValueError, "subs_nz should have at least 3 elements.")

    D, K = 0, None
    out: np.ndarray = np.array([])

    if m < 3:
        D, K = u.shape if m == 1 else v.shape
        out = np.zeros((D, K))
    else:
        log_and_raise_error(ValueError, "m should be 1 or 2.")

    if K is not None:
        for k in range(K):
            # Select the appropriate indices
            indices = subs[0] if temporal else 0
            # Copy the values to avoid modifying the original array
            copied_vals = vals.copy()
            # Select the appropriate slice of the affinity tensor
            w_I = w[indices, k, :] if m == 1 else w[indices, :, k]
            # Ensure that the affinity tensor slice has the same data type as the values
            w_I = (w_I if temporal else w_I[np.newaxis, :]).astype(copied_vals.dtype)
            # Select the appropriate slice of the membership matrix
            multiplier_u_v = v[subs[2], :] if m == 1 else u[subs[1], :]
            # Ensure that the membership matrix slice has the same data type as the values
            multiplier_u_v = (
                multiplier_u_v.astype(copied_vals.dtype) if temporal else multiplier_u_v
            )
            # Multiply the values by the affinity tensor slice and the membership matrix slice
            copied_vals *= (w_I * multiplier_u_v).sum(axis=1)
            # Update the evaluation matrix with the weighted sum of the values
            out[:, k] += np.bincount(subs[m], weights=copied_vals, minlength=D)

    return out


def normalize_nonzero_membership(u: np.ndarray, axis: Optional[int] = 1) -> np.ndarray:
    """
    Given a matrix, it returns the same matrix normalized by row.

    Parameters
    ----------
    u: ndarray
       Numpy Matrix.
    axis: Optional[int]
          Axis along which the matrix should be normalized.

    Returns
    -------
    The matrix normalized by row.
    """

    # Calculate the sum of elements along axis 1, keeping the same dimensions.
    den1 = u.sum(axis=axis, keepdims=True)

    # Identify the positions where den1 is equal to 0 and create a boolean mask.
    nzz = den1 == 0.0

    # Replace the elements in den1 corresponding to positions where it is 0
    # with 1 to avoid division by zero.
    den1[nzz] = 1.0

    # Normalize the matrix u by dividing each element by the corresponding sum along axis 1.
    return u / den1


def transpose_matrix(M: np.ndarray) -> np.ndarray:
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
    # Return the transpose of a matrix
    return np.einsum("ij->ji", M)


def transpose_tensor(M: np.ndarray) -> np.ndarray:
    """
    Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.

    Parameters
    ----------
    M : ndarray
        Tensor with the mean lambda for all entries.

    Returns
    -------
    Transpose version of M_aij, i.e. M_aji.
    """

    return np.einsum("aij->aji", M)


def Exp_ija_matrix(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
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

    # Compute the outer product of matrices u and v, resulting in a 4D tensor M.
    # Dimensions of M: (number of rows in u) x (number of columns in v) x
    # (number of rows in u) x (number of columns in v)
    M = np.einsum("ik,jq->ijkq", u, v)

    # Multiply the 4D tensor M element-wise with the 2D tensor w along the last dimension.
    # Dimensions of w: (number of columns in v) x (number of columns in w)
    # Resulting tensor after the einsum operation: (number of rows in u) x (number of rows in u)
    M = np.einsum("ijkq,kq->ij", M, w)

    return M
