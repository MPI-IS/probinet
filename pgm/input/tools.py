"""
It provides utility functions to handle matrices, tensors, and sparsity. It includes functions for
checking if an object can be cast to an integer, normalizing matrices, determining sparsity in
tensors, and converting between dense and sparse representations.
"""
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import sktensor as skt


def can_cast(string: Union[int, float, str]) -> bool:
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


def normalize_nonzero_membership(u: np.ndarray) -> np.ndarray:
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

    # Calculate the sum of elements along axis 1, keeping the same dimensions.
    den1 = u.sum(axis=1, keepdims=True)

    # Identify the positions where den1 is equal to 0 and create a boolean mask.
    nzz = den1 == 0.

    # Replace the elements in den1 corresponding to positions where it is 0
    # with 1 to avoid division by zero.
    den1[nzz] = 1.

    # Normalize the matrix u by dividing each element by the corresponding sum along axis 1.
    return u / den1


def is_sparse(X: np.ndarray) -> bool:
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

    # Get the number of dimensions of the input tensor X.
    M = X.ndim

    # Get the total number of elements in the tensor X.
    S = X.size

    # Get the number of non-zero elements in the tensor X using the first
    # dimension of the non-zero indices.
    I = X.nonzero()[0].size

    # Check if the tensor X is sparse based on a heuristic definition of sparsity.
    # A tensor is considered sparse if the total number of elements is greater
    # than (number of non-zero elements + 1) times the number of dimensions.
    return S > (I + 1) * M


def sptensor_from_dense_array(X: np.ndarray) -> skt.sptensor:
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

    # Extract the values of non-zero elements in the dense tensor X.
    vals = X[subs]

    # Create a sparse tensor (sptensor) using the non-zero indices (subs) and corresponding
    # values (vals). The shape and data type of the sparse tensor are derived
    # from the original dense tensor X.
    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def get_item_array_from_subs(A: np.ndarray, ref_subs: Tuple[np.ndarray]) -> np.ndarray:
    """
    Get values of ref_subs entries of a dense tensor.
    Output is a 1-d array with dimension = number of non zero entries.
    """

    # return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)]) (Older
    # version) #TODO: Check with Martina
    return np.array([A[tuple(sub)] for sub in zip(*ref_subs)])


# TODO: merge into one
def transpose_ij3(M: np.ndarray) -> np.ndarray:
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
    # Return the transpose of a tensor
    return np.einsum('aij->aji', M)


def transpose_ij2(M: np.ndarray) -> np.ndarray:
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
    return np.einsum('ij->ji', M)


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
    M = np.einsum('ik,jq->ijkq', u, v)

    # Multiply the 4D tensor M element-wise with the 2D tensor w along the last dimension.
    # Dimensions of w: (number of columns in v) x (number of columns in w)
    # Resulting tensor after the einsum operation: (number of rows in u) x (number of rows in u)
    M = np.einsum('ijkq,kq->ij', M, w)

    return M


def check_symmetric(a: Union[np.ndarray, List[np.ndarray]],
                    rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Check if a matrix or a list of matrices is symmetric.

    Parameters
    ----------
    a : ndarray or list
        Input data.
    rtol : float
           Relative tolerance.
    atol : float
              Absolute tolerance.
    Returns
    -------
    True if the matrix is symmetric, False otherwise.
    """
    if isinstance(a, list):
        return all(np.allclose(mat, mat.T, rtol=rtol, atol=atol) for mat in a)

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def build_edgelist(A: skt.sptensor, l: int) -> pd.DataFrame:
    """
    Build the edgelist for a given layer, a in DataFrame format.

    Parameters
    ----------
    A : ndarray or sptensor
        List of scipy sparse matrices, one for each layer
    l : int
        Layer index.
    Returns
    -------
    Dataframe with the edgelist for a given layer.
    """

    A_coo = A.tocoo()  # TODO: Solve this issue with Martina
    data_dict = {'source': A_coo.row, 'target': A_coo.col, 'L' + str(l): A_coo.data}
    df_res = pd.DataFrame(data_dict)

    return df_res


def output_adjacency(A: List, out_folder: str, label: str):
    """
    Save the adjacency tensor to a file.
    Default format is space-separated .csv with L+2 columns: source_node target_node
    edge_l0 ... edge_lL .

    Parameters
    ----------
    A : ndarray
        Adjacency tensor.
    out_folder : str
        Output folder.
    label : str
        Label of the output file.
    """

    outfile = label + '.dat'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    L = len(A)
    df = pd.DataFrame()
    for l in range(L):
        dfl = build_edgelist(A[l], l)
        df = df.append(dfl)
    df.to_csv(out_folder + outfile, index=False, sep=' ')
    print(f'Adjacency matrix saved in: {out_folder + outfile}')


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

    return np.einsum('aij->aji', M)
