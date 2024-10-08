"""
It provides utility functions to handle matrices, tensors, and sparsity. It includes functions for
checking if an object can be cast to an integer, normalizing matrices, determining sparsity in
tensors, and converting between dense and sparse representations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import networkx as nx
import numpy as np
import pandas as pd
from sparse import COO


def can_cast_to_int(string: Union[int, float, str]) -> bool:
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
        logging.error("Cannot cast %s to integer.", string)
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
    nzz = den1 == 0.0

    # Replace the elements in den1 corresponding to positions where it is 0
    # with 1 to avoid division by zero.
    den1[nzz] = 1.0

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


def sptensor_from_dense_array(X: np.ndarray) -> COO:
    """
    Create a sparse tensor from a dense array using sparse.COO.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    COO
        Sparse tensor created from the dense array.
    """
    # Get the non-zero indices and values from the dense array
    coords = np.array(X.nonzero())
    data = X[tuple(coords)].astype(float)

    # Create the sparse tensor using COO format
    sparse_X = COO(coords, data, shape=X.shape)
    return sparse_X


def get_item_array_from_subs(A: np.ndarray, ref_subs: Tuple[np.ndarray]) -> np.ndarray:
    """
    Retrieves the values of specific entries in a dense tensor.
    Output is a 1-d array with dimension = number of non-zero entries.

    Parameters
    ----------
    A : np.ndarray
        The input tensor from which values are to be retrieved.

    ref_subs : Tuple[np.ndarray]
        A tuple containing arrays of indices. Each array in the tuple corresponds to indices along
        one dimension of the tensor.

    Returns
    -------
    np.ndarray
        A 1-dimensional array containing the values of the tensor at the specified indices.
    """
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
    return np.einsum("aij->aji", M)


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
    return np.einsum("ij->ji", M)


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


def check_symmetric(
    a: Union[np.ndarray, List[np.ndarray]], rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """
    Check if a matrix or a list of matrices is symmetric.

    Parameters
    ----------
    a : ndarray or list
        Input data.
    rtol : float
           Relative convergence_tol.
    atol : float
           Absolute convergence_tol.

    Returns
    -------
    True if the matrix is symmetric, False otherwise.
    """

    if isinstance(a, list):
        return all(np.allclose(mat, mat.T, rtol=rtol, atol=atol) for mat in a)

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def build_edgelist(A: COO, layer: int) -> pd.DataFrame:
    """
    Build the edgelist for a given layer of an adjacency tensor in DataFrame format.

    Parameters
    ----------
    A : coo_matrix
        Sparse matrix in COOrdinate format representing the adjacency tensor.
    layer : int
        Index of the layer for which the edgelist is to be built.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the edgelist for the specified layer with columns 'source', 'target', and 'L<layer>'.
    """

    # Convert the input sparse matrix A to COOrdinate format
    A_coo = A.tocoo()

    # Create a dictionary with 'source', 'target', and 'L' keys
    # 'source' and 'target' represent the row and column indices of non-zero elements in A
    # 'L' represents the data of non-zero elements in A
    data_dict = {"source": A_coo.row, "target": A_coo.col, "L" + str(layer): A_coo.data}

    # Convert the dictionary to a pandas DataFrame
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

    # Concatenate the label with '.dat' to form the output file name
    outfile = label + ".dat"

    # Create a Path object for the output folder
    out_folder_path = Path(out_folder)

    # Create the output directory. If the directory already exists, no exception is raised.
    # parents=True ensures that any missing parent directories are also created.
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # For each layer in A, build an edge list and store them in a list
    df_list = [build_edgelist(A[layer], layer) for layer in range(len(A))]

    # Concatenate all the DataFrames in the list into a single DataFrame
    df = pd.concat(df_list)

    # Save the DataFrame to a CSV file in the output directory
    # index=False prevents pandas from writing row indices in the CSV file
    # sep=' ' specifies that the fields are separated by a space
    df.to_csv(out_folder + outfile, index=False, sep=" ")

    # Print the location where the adjacency matrix is saved
    logging.info("Adjacency matrix saved in: %s", out_folder + outfile)


def write_adjacency(
    G: List[nx.MultiDiGraph],
    folder: str = "./",
    fname: str = "adj.csv",
    ego: str = "source",
    alter: str = "target",
):
    """
    Save the adjacency tensor to file.

    Parameters
    ----------
    G : list
        List of MultiDiGraph NetworkX objects.
    folder : str
             Path of the folder where to save the files.
    fname : str
            Name of the adjacency tensor file.
    ego : str
          Name of the column to consider as source of the edge.
    alter : str
            Name of the column to consider as target of the edge.
    """

    N = G[0].number_of_nodes()
    L = len(G)
    B = np.empty(shape=[len(G), N, N])
    for layer in range(len(G)):
        B[layer, :, :] = nx.to_numpy_array(G[layer], weight="weight")
    df_list = []
    for i in range(N):
        for j in range(N):
            Z = 0
            for layer in range(L):
                Z += B[layer][i][j]
            if Z > 0:
                data = [i, j]
                data.extend([int(B[a][i][j]) for a in range(L)])
                df_list.append(data)
    cols = [ego, alter]
    cols.extend(["L" + str(l) for l in range(1, L + 1)])
    df = pd.DataFrame(df_list, columns=cols)
    df.to_csv(path_or_buf=folder + fname, index=False)
    logging.info("Adjacency tensor saved in: %s", folder + fname)


def write_design_Matrix(
    metadata: Dict[str, str],
    perc: float,
    folder: str = "./",
    fname: str = "X_",
    nodeID: str = "Name",
    attr_name: str = "Metadata",
):
    """
    Save the design matrix to file.

    Parameters
    ----------
    metadata : dict
               Dictionary where the keys are the node labels and the values are the metadata associated to them.
    perc : float
           Fraction of match between communities and metadata.
    folder : str
             Path of the folder where to save the files.
    fname : str
            Name of the design matrix file.
    nodeID : str
             Name of the column with the node labels.
    attr_name : str
                Name of the column to consider as attribute.
    """

    # Create a DataFrame from the metadata dictionary
    X = pd.DataFrame.from_dict(metadata, orient="index", columns=[attr_name])

    # Create a new column with the node labels
    X[nodeID] = X.index

    # Select the columns in the order specified by nodeID and attr_name
    X = X.loc[:, [nodeID, attr_name]]

    # Construct the file path using f-string formatting and Path
    file_path = Path(folder) / f"{fname}{str(perc)[0]}_{str(perc)[2]}.csv"

    # Save the DataFrame to a CSV file
    X.to_csv(path_or_buf=file_path, index=False)

    # Log the location where the design matrix is saved
    logging.debug("Design matrix saved in: %s", file_path)


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
            # Update the output matrix with the weighted sum of the values
            out[:, k] += np.bincount(subs[m], weights=copied_vals, minlength=D)

    return out


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

    # Determine the shape of the output array
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
        # Update the output matrix with the weighted sum of the values
        out[:, k] += np.bincount(subs[m], weights=copied_vals, minlength=D)

    return out


def log_and_raise_error(error_type: Type[BaseException], message: str) -> None:
    """
    Logs an error message and raises an exception of the specified type.

    Parameters
    ----------
    error_type : Type[BaseException]
        The type of the exception to be raised.
    message : str
        The error message to be logged and included in the exception.

    Raises
    ------
    BaseException
        An exception of the specified type with the given message.
    """

    # Log the error message
    logging.error(message)

    # Raise the exception
    raise error_type(message)


def flt(x: float, d: int = 3) -> float:
    """
    Round a number to a specified number of decimal places.

    Parameters
    ----------
    x : float
        Number to be rounded.
    d : int
        Number of decimal places to round to.
    Returns
    -------
    float
        The input number rounded to the specified number of decimal places.
    """
    return round(x, d)


def transpose_ij(M: np.ndarray) -> np.ndarray:
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
    return np.einsum("aij->aji", M)


def inherit_docstring(cls, from_init=True):
    """
    Decorator to inherit the docstring from the parent class.

    Parameters
    ----------
    cls : class
        The parent class from which to inherit the docstring.
    from_init : bool, optional
        If True, inherit the docstring from the __init__ method of the parent class.
        If False, inherit the class-level docstring from the parent class.
    """

    # Based on the implementation from the scikit-tensor library:
    # https://github.com/mnick/scikit-tensor/blob/master/sktensor/pyutils.py

    def decorator(func):
        if from_init:
            # Inherit the docstring from the __init__ method
            parent_init = getattr(cls, "__init__", None)
            if parent_init and parent_init.__doc__:
                func.__doc__ = parent_init.__doc__
        else:
            # Inherit the class-level docstring
            if class_docstring := cls.__doc__:
                func.__doc__ = class_docstring

        return func

    return decorator
