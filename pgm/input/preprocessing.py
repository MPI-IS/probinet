"""
It provides functions for preprocessing and constructing adjacency tensors from NetworkX graphs.
The script facilitates the creation of both dense and sparse adjacency tensors, considering edge
weights, and ensures proper formatting of input data tensors.
"""

from typing import Any, List, Optional, Tuple, Union

import networkx as nx
from numpy import ndarray
import numpy as np
import scipy
from sktensor import sptensor
import sktensor as skt

from . import tools


def build_B_from_A(A: List[nx.MultiDiGraph], nodes: Optional[List] = None,
                   calculate_reciprocity: bool = True) -> Union[ndarray, Tuple[ndarray, List[Any]]]:
    """
    Create the numpy adjacency tensor of a networkX graph.

    Parameters
    ----------
    A : list
        List of MultiDiGraph NetworkX objects.
    nodes : list, optional
        List of nodes IDs. If not provided, use all nodes in the first graph as the default.
    calculate_reciprocity : bool, optional
        Whether to calculate reciprocity or not. Default is True.

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
    for l, A_layer in enumerate(A):
        # Check if the current graph has the same set of nodes as the first graph
        assert set(A_layer.nodes()) == set(
            nodes), "All graphs in A must have the same set of vertices."

        # Check if all weights in B[l] are integers
        assert all(isinstance(a[2], int) for a in
                   A_layer.edges(data='weight')), "All weights in A must be integers."

        # Convert the graph A[l] to a numpy array with specified options
        # - weight='weight': consider edge weights
        # - dtype=int: ensure the resulting array has integer data type
        # - nodelist=nodes: use the specified nodes
        B[l, :, :] = nx.to_numpy_array(A_layer,
                                       weight='weight',
                                       dtype=int,
                                       nodelist=nodes)

        # Calculate reciprocity for the current layer and append it to the rw list
        if calculate_reciprocity:
            rw_layer = np.multiply(B[l], B[l].T).sum() / B[l].sum()
            rw.append(rw_layer)

    if not calculate_reciprocity:
        rw = []

    return B, rw


def build_sparse_B_from_A(A: List[nx.MultiDiGraph],
                          calculate_reciprocity: bool = False) -> Union[Tuple[sptensor,
                                                                              sptensor,
                                                                              ndarray,
                                                                              List[Any]], sptensor]:
    """
    Create the sptensor adjacency tensor of a networkX graph.

    Parameters
    ----------
    A : list
        List of MultiDiGraph NetworkX objects.
    calculate_reciprocity : bool, optional
        Whether to calculate and return the reciprocity values. Default is False.

    Returns
    -------
    data : sptensor or Tuple[sptensor, sptensor, ndarray, List[Any]]
           Graph adjacency tensor. If calculate_reciprocity is True, returns a tuple with the adjacency tensor, its transpose, an array with values of entries A[j, i] given non-zero entry (i, j), and a list of reciprocity values.
    """

    # Get the number of nodes in the first graph of the list A
    N = A[0].number_of_nodes()

    # Get the number of layers (graphs) in A
    L = len(A)

    # Initialize an empty list to store reciprocity values for each layer in A
    rw = []

    # Initialize arrays to store indices and values for building sparse tensors
    d1 = np.array((), dtype='int64')
    d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    v, vT, v_T = np.array(()), np.array(()), np.array(())  # type: ndarray, ndarray, ndarray

    # Loop over each layer in A
    for l in range(L):
        # Convert the graph A[l] to a scipy sparse array and its transpose
        b = nx.to_scipy_sparse_array(A[l])
        b_T = nx.to_scipy_sparse_array(A[l]).transpose()

        # Calculate reciprocity for the current layer and append it to the rw list
        if calculate_reciprocity:
            rw.append(np.sum(b.multiply(b_T)) / np.sum(b))

        # Get the non-zero indices for the original and transposed arrays
        nz = b.nonzero()
        nz_T = b_T.nonzero()

        # Append indices and values to the arrays for building sparse tensors
        d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
        d2 = np.hstack((d2, nz[0]))
        d2_T = np.hstack((d2_T, nz_T[0]))
        d3 = np.hstack((d3, nz[1]))
        d3_T = np.hstack((d3_T, nz_T[1]))
        v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
        vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
        v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))

    # Create sparse tensors for the original and transposed graphs
    subs_ = (d1, d2, d3)
    subs_T_ = (d1, d2_T, d3_T)
    data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
    data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)

    if calculate_reciprocity:
        return data, data_T, v_T, rw
    else:
        return data


def preprocess(A: np.ndarray) -> sptensor:
    """
    Pre-process input data tensor.
    If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

    Parameters
    ----------
    A : ndarray
        Input data (tensor).

    Returns
    -------
    A : sptensor/dtensor
        Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns
        an int dtensor.
    """

    if not A.dtype == np.dtype(int).type:
        A = A.astype(int)
    if np.logical_and(isinstance(A, np.ndarray), tools.is_sparse(A)):
        A = tools.sptensor_from_dense_array(A)
    else:
        A = skt.dtensor(A)

    return A


def preprocess_X(X):
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
