""" Functions for handling and visualizing the data. """

# import networkx as nx
# import numpy as np
# import pandas as pd
# import sktensor as skt
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import seaborn as sns
# sns.set_style('white')



# this one is equal to the one in pgm/input/preprocessing.py
# def build_B_from_A(A, nodes=None):
#     """
#         Create the numpy adjacency tensor of a networkX graph.
#
#         Parameters
#         ----------
#         A : list
#             List of MultiDiGraph NetworkX objects.
#         nodes : list
#                 List of nodes IDs.
#
#         Returns
#         -------
#         B : ndarray
#             Graph adjacency tensor.
#         rw : list
#              List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
#     """
#
#     N = A[0].number_of_nodes()
#     if nodes is None:
#         nodes = list(A[0].nodes())
#     B = np.empty(shape=[len(A), N, N])
#     rw = []
#     for l in range(len(A)):
#         B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes)
#         rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())
#
#     return B, rw

# this is the same as in pgm/input/preprocessing.py
# def build_sparse_B_from_A(A):
#     """
#         Create the sptensor adjacency tensor of a networkX graph.
#
#         Parameters
#         ----------
#         A : list
#             List of MultiDiGraph NetworkX objects.
#
#         Returns
#         -------
#         data : sptensor
#                Graph adjacency tensor.
#         data_T : sptensor
#                  Graph adjacency tensor (transpose).
#         v_T : ndarray
#               Array with values of entries A[j, i] given non-zero entry (i, j).
#         rw : list
#              List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
#     """
#
#     N = A[0].number_of_nodes()
#     L = len(A)
#     rw = []
#
#     d1 = np.array((), dtype='int64')
#     d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
#     d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
#     v, vT, v_T = np.array(()), np.array(()), np.array(())
#     for l in range(L):
#         b = nx.to_scipy_sparse_array(A[l])
#         b_T = nx.to_scipy_sparse_array(A[l]).transpose()
#         rw.append(np.sum(b.multiply(b_T)) / np.sum(b))
#         nz = b.nonzero()
#         nz_T = b_T.nonzero()
#         d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
#         d2 = np.hstack((d2, nz[0]))
#         d2_T = np.hstack((d2_T, nz_T[0]))
#         d3 = np.hstack((d3, nz[1]))
#         d3_T = np.hstack((d3_T, nz_T[1]))
#         v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
#         vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
#         v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
#     subs_ = (d1, d2, d3)
#     subs_T_ = (d1, d2_T, d3_T)
#     data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
#     data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)
#
#     return data, data_T, v_T, rw

# same as in pgm/input/statistics.py
# def reciprocal_edges(G):
#     """
#         Compute the proportion of bi-directional edges, by considering the unordered pairs.
#
#         Parameters
#         ----------
#         G: MultiDigraph
#            MultiDiGraph NetworkX object.
#
#         Returns
#         -------
#         reciprocity: float
#                      Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
#     """
#
#     n_all_edge = G.number_of_edges()
#     n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
#     n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network
#
#     if n_all_edge == 0:
#         raise nx.NetworkXError("Not defined for empty graphs.")
#
#     reciprocity = float(n_overlap_edge) / float(n_undirected)
#
#     return reciprocity

# this is the same as in pgm/input/tools.py
# def normalize_nonzero_membership(U):
#     """
#         Given a matrix, it returns the same matrix normalized by row.
#
#         Parameters
#         ----------
#         U: ndarray
#            Numpy Matrix.
#
#         Returns
#         -------
#         The matrix normalized by row.
#     """
#
#     den1 = U.sum(axis=1, keepdims=True)
#     nzz = den1 == 0.
#     den1[nzz] = 1.
#
#     return U / den1

# this is the same as in pgm/input/tools.py but has a different name. There it is called transpose_ij3()
# def transpose_tensor(M):
#     """
#         Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.
# 
#         Parameters
#         ----------
#         M : ndarray
#             Tensor with the mean lambda for all entries.
# 
#         Returns
#         -------
#         Transpose version of M_aij, i.e. M_aji.
#     """
# 
#     return np.einsum('aij->aji', M)




# this is in pgm/output/evaluate.py
# def lambda0_full(u, v, w):
#     """
#         Compute the mean lambda0 for all entries.
#
#         Parameters
#         ----------
#         u : ndarray
#             Out-going membership matrix.
#         v : ndarray
#             In-coming membership matrix.
#         w : ndarray
#             Affinity tensor.
#
#         Returns
#         -------
#         M : ndarray
#             Mean lambda0 for all entries.
#     """
#
#     if w.ndim == 2:
#         M = np.einsum('ik,jk->ijk', u, v)
#         M = np.einsum('ijk,ak->aij', M, w)
#     else:
#         M = np.einsum('ik,jq->ijkq', u, v)
#         M = np.einsum('ijkq,akq->aij', M, w)
#
#     return M



