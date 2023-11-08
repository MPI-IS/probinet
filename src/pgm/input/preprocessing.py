"""
It provides functions for preprocessing and constructing adjacency tensors from NetworkX graphs. The script facilitates
the creation of both dense and sparse adjacency tensors, considering edge weights, and ensures proper formatting of
input data tensors.
"""
import networkx as nx
import numpy as np
import sktensor as skt

from . import tools


def build_B_from_A(A, nodes=None):
    """
        Create the numpy adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        nodes : list
                List of nodes IDs.

        Returns
        -------
        B : ndarray
            Graph adjacency tensor.
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    if nodes is None:
        nodes = list(A[0].nodes())
    B = np.empty(shape=[len(A), N, N])
    rw = []
    for l in range(len(A)):
        B[l, :, :] = nx.to_numpy_array(A[l],
                                       weight='weight',
                                       dtype=int,
                                       nodelist=nodes)
        rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())

    return B, rw


def build_sparse_B_from_A(A):
    """
        Create the sptensor adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        data : sptensor
               Graph adjacency tensor.
        data_T : sptensor
                 Graph adjacency tensor (transpose).
        v_T : ndarray
              Array with values of entries A[j, i] given non-zero entry (i, j).
        rw : list
             List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
    """

    N = A[0].number_of_nodes()
    L = len(A)
    rw = []

    d1 = np.array((), dtype='int64')
    d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
    v, vT, v_T = np.array(()), np.array(()), np.array(())
    for l in range(L):
        b = nx.to_scipy_sparse_array(A[l])
        b_T = nx.to_scipy_sparse_array(A[l]).transpose()
        rw.append(np.sum(b.multiply(b_T)) / np.sum(b))
        nz = b.nonzero()
        nz_T = b_T.nonzero()
        d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
        d2 = np.hstack((d2, nz[0]))
        d2_T = np.hstack((d2_T, nz_T[0]))
        d3 = np.hstack((d3, nz[1]))
        d3_T = np.hstack((d3_T, nz_T[1]))
        v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
        vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
        v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
    subs_ = (d1, d2, d3)
    subs_T_ = (d1, d2_T, d3_T)
    data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
    data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)

    return data, data_T, v_T, rw


def preprocess(A):
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
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not A.dtype == np.dtype(int).type:
        A = A.astype(int)
    if np.logical_and(isinstance(A, np.ndarray), tools.is_sparse(A)):
        A = tools.sptensor_from_dense_array(A)
    else:
        A = skt.dtensor(A)

    return A
