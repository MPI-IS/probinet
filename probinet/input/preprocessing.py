"""
It provides functions for preprocessing and constructing adjacency tensors from NetworkX graphs.
The script facilitates the creation of both dense and sparse adjacency tensors, considering edge
weights, and ensures proper formatting of input data tensors.
"""

from typing import Any, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy
from numpy import ndarray
from sparse import COO

from ..types import GraphDataType
from ..utils import tools


def create_adjacency_tensor_from_graph_list(
    A: List[nx.MultiDiGraph],
    nodes: Optional[List] = None,
    calculate_reciprocity: bool = True,
    label: str = "weight",
) -> Union[ndarray, Tuple[ndarray, List[Any]]]:
    """
    Create the numpy adjacency tensor of a networkX graph.

    Parameters
    ----------
    A : list
        List of MultiDiGraph NetworkX objects.
    nodes : list, optional
            List of nodes IDs. If not provided, use all nodes in the first graph as the default.
    calculate_reciprocity : bool, optional
                            Whether to calculate reciprocity or not.
    label : str, optional
            The edge attribute key used to determine the weight of the edges.

    Returns
    -------
    B : ndarray or Tuple[ndarray, List[Any]]
        Graph adjacency tensor. If calculate_reciprocity is True, returns a tuple with B and a list of reciprocity values.

    Raises
    ------
    AssertionError
        If any graph in A has a different set of vertices than the first graph.
        If any weight in B is not an integer.
    """

    # Get the number of nodes in the first graph of the list A
    N = A[0].number_of_nodes()

    # If nodes is not provided, use all nodes in the first graph as the default
    if nodes is None:
        nodes = list(A[0].nodes())

    # Initialize an empty numpy array B to store the adjacency tensors for each layer in A
    # The shape of B is [number of layers in A, N, N]
    B = np.empty(shape=[len(A), N, N])

    # Initialize an empty list to store reciprocity values for each layer in A
    rw = []

    # Loop over each layer in A using enumeration to get both the layer index (l) and the graph
    # (A[l])
    for layer, A_layer in enumerate(A):
        # Check if the current graph has the same set of nodes as the first graph
        assert set(A_layer.nodes()) == set(
            nodes
        ), "All graphs in A must have the same set of vertices."

        # Check if all weights in B[l] are integers
        assert all(
            isinstance(a[2], int) for a in A_layer.edges(data=label)
        ), "All weights in A must be integers."

        # Convert the graph A[l] to a numpy array with specified options
        # - weight='weight': consider edge weights
        # - dtype=int: ensure the resulting array has integer data type
        # - nodelist=nodes: use the specified nodes
        B[layer, :, :] = nx.to_numpy_array(
            A_layer, weight=label, dtype=int, nodelist=nodes
        )

        # Calculate reciprocity for the current layer and append it to the rw list
        if calculate_reciprocity:
            rw_layer = np.multiply(B[layer], B[layer].T).sum() / B[layer].sum()
            rw.append(rw_layer)

    if not calculate_reciprocity:
        rw = []

    return B, rw


def create_sparse_adjacency_tensor_from_graph_list(
    A: List[nx.MultiDiGraph], calculate_reciprocity: bool = False
) -> Union[COO, Tuple[COO, COO, ndarray, List[Any]]]:
    """
    Create the sparse tensor adjacency tensor of a networkX graph using TensorLy.

    Parameters
    ----------
    A : list
        List of MultiDiGraph NetworkX objects.
    calculate_reciprocity : bool, optional
                            Whether to calculate and return the reciprocity values..

    Returns
    -------
    data : SparseTensor or Tuple[SparseTensor, SparseTensor, ndarray, List[Any]]
           Graph adjacency tensor. If calculate_reciprocity is True, returns a tuple with the adjacency tensor, its transpose, an array with values of entries A[j, i] given non-zero entry (i, j), and a list of reciprocity values.
    """

    # Get the number of nodes in the first graph of the list A
    N = A[0].number_of_nodes()

    # Get the number of layers (graphs) in A
    L = len(A)

    # Initialize an empty list to store reciprocity values for each layer in A
    rw = []

    # Initialize arrays to store indices and values for building sparse tensors
    d1 = []
    d2, d2_T = [], []
    d3, d3_T = [], []
    v, vT, v_T = [], [], []

    # Loop over each layer in A
    for layer in range(L):
        # Convert the graph A[layer] to a scipy sparse array and its transpose
        b = nx.to_scipy_sparse_array(A[layer])
        b_T = b.transpose()

        # Calculate reciprocity for the current layer and append it to the rw list
        if calculate_reciprocity:
            rw.append(np.sum(b.multiply(b_T)) / np.sum(b))

        # Get the non-zero indices for the original and transposed arrays
        nz = b.nonzero()
        nz_T = b_T.nonzero()

        # Append indices and values to the arrays for building sparse tensors
        d1.extend([layer] * len(nz[0]))
        d2.extend(nz[0])
        d2_T.extend(nz_T[0])
        d3.extend(nz[1])
        d3_T.extend(nz_T[1])
        v.extend(b[nz])
        vT.extend(b_T[nz_T])
        v_T.extend(b[nz[::-1]])

    # Create sparse tensors for the original and transposed graphs
    subs_ = (np.array(d1), np.array(d2), np.array(d3))
    subs_T_ = (np.array(d1), np.array(d2_T), np.array(d3_T))
    data = COO(subs_, np.array(v, dtype=np.float64), shape=(L, N, N))
    data_T = COO(subs_T_, np.array(vT, dtype=np.float64), shape=(L, N, N))

    if calculate_reciprocity:
        return data, data_T, np.array(v_T), rw
    return data


def preprocess_adjacency_tensor(A: np.ndarray) -> GraphDataType:
    """
    Pre-process input data tensor.

    If the input is sparse, returns an integer sparse tensor (COO). Otherwise, returns an integer dense tensor (ndarray).

    Parameters
    ----------
    A : ndarray
        Input data tensor.

    Returns
    -------
    A : COO or ndarray
        Pre-processed data. If the input is sparse, returns an integer sparse tensor (COO). Otherwise, returns an integer dense tensor (ndarray).
    """
    if not A.dtype == np.dtype(int).type:
        A = A.astype(int)
    if np.logical_and(isinstance(A, np.ndarray), tools.is_sparse(A)):
        A = tools.sptensor_from_dense_array(A)

    return A


def preprocess_data_matrix(X):
    """
    Pre-process input data matrix.
    If the input is sparse, returns a scipy sparse matrix. Otherwise, returns a numpy array.

    Parameters
    ----------
    X : ndarray
        Input data (matrix).

    Returns
    -------
    X : scipy sparse matrix/ndarray
        Pre-processed data. If the input is sparse, returns a scipy sparse matrix. Otherwise, returns a numpy array.
    """

    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if np.logical_and(isinstance(X, np.ndarray), scipy.sparse.issparse(X)):
        X = scipy.sparse.csr_matrix(X)

    return X
