"""
It provides utility functions to handle matrices, tensors, and sparsity. It includes functions for
checking if an object can be cast to an integer, normalizing matrices, determining sparsity in
tensors, and converting between dense and sparse representations.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
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
        logging.error('Cannot cast %s to integer.', string)
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


def build_edgelist(A: coo_matrix, l: int) -> pd.DataFrame:
    """
    Build the edgelist for a given layer, a in DataFrame format.

    Parameters
    ----------
    A : ndarray or sptensor
        Adjacency tensor.
    l : int
        Layer index.
    Returns
    -------
    Dataframe with the edgelist for a given layer.
    """

    # Convert the input sparse matrix A to COOrdinate format
    A_coo = A.tocoo()

    # Create a dictionary with 'source', 'target', and 'L' keys
    # 'source' and 'target' represent the row and column indices of non-zero elements in A
    # 'L' represents the data of non-zero elements in A
    data_dict = {'source': A_coo.row, 'target': A_coo.col, 'L' + str(l): A_coo.data}

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
    outfile = label + '.dat'

    # Create a Path object for the output folder
    out_folder_path = Path(out_folder)

    # Create the output directory. If the directory already exists, no exception is raised.
    # parents=True ensures that any missing parent directories are also created.
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # For each layer in A, build an edge list and store them in a list
    df_list = [build_edgelist(A[l], l) for l in range(len(A))]

    # Concatenate all the DataFrames in the list into a single DataFrame
    df = pd.concat(df_list)

    # Save the DataFrame to a CSV file in the output directory
    # index=False prevents pandas from writing row indices in the CSV file
    # sep=' ' specifies that the fields are separated by a space
    df.to_csv(out_folder + outfile, index=False, sep=' ')

    # Print the location where the adjacency matrix is saved
    logging.info('Adjacency matrix saved in: %s', out_folder + outfile)


def write_adjacency(G: List[nx.MultiDiGraph],
                    folder: str = './',
                    fname: str = 'adj.csv',
                    ego: str = 'source',
                    alter: str = 'target'):
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
    for l in range(len(G)):
        B[l, :, :] = nx.to_numpy_array(G[l], weight='weight')
    df_list = []
    for i in range(N):
        for j in range(N):
            Z = 0
            for l in range(L):
                Z += B[l][i][j]
            if Z > 0:
                data = [i, j]
                data.extend([int(B[a][i][j]) for a in range(L)])
                df_list.append(data)
    cols = [ego, alter]
    cols.extend(['L' + str(l) for l in range(1, L + 1)])
    df = pd.DataFrame(df_list, columns=cols)
    df.to_csv(path_or_buf=folder + fname, index=False)
    logging.info('Adjacency tensor saved in: %s', folder + fname)


def write_design_Matrix(
        metadata: Dict[str, str],
        perc: float,
        folder: str = './',
        fname: str = 'X_',
        nodeID: str = 'Name',
        attr_name: str = 'Metadata'):
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
    X = pd.DataFrame.from_dict(metadata, orient='index', columns=[attr_name])

    # Create a new column with the node labels
    X[nodeID] = X.index

    # Select the columns in the order specified by nodeID and attr_name
    X = X.loc[:, [nodeID, attr_name]]

    # Construct the file path using f-string formatting and Path
    file_path = Path(folder) / f"{fname}{str(perc)[0]}_{str(perc)[2]}.csv"

    # Save the DataFrame to a CSV file
    X.to_csv(path_or_buf=file_path, index=False)

    # Log the location where the design matrix is saved
    logging.debug(f'Design matrix saved in: {file_path}')

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


def sp_uttkrp(vals: np.ndarray, subs: Tuple[np.ndarray], m: int, u: np.ndarray,
              v: np.ndarray, w: np.ndarray, temporal: bool = True) -> np.ndarray:
    """
    Compute the Khatri-Rao product (sparse version).

    Parameters
    ----------
    vals : ndarray
           Values of the non-zero entries.
    subs : tuple
           Indices of elements that are non-zero. It is a n-tuple of array-likes and the length
           of tuple n must be
           equal to the dimension of tensor.
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

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if temporal:
            if m == 1:  # we are updating u
                tmp *= (w[subs[0], k, :].astype(tmp.dtype) *
                        v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
            elif m == 2:  # we are updating v
                tmp *= (w[subs[0], :, k].astype(tmp.dtype) *
                        u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        else:
            if m == 1:  # we are updating u
                w_I = w[0, k, :]
                tmp *= (w_I[np.newaxis, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(
                    axis=1)
            elif m == 2:  # we are updating v
                w_I = w[0, :, k]
                tmp *= (w_I[np.newaxis, :].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(
                    axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals: np.ndarray, subs: Tuple[np.ndarray], m: int,
                          u: np.ndarray, v: np.ndarray,
                          w: np.ndarray, temporal: bool = False) -> np.ndarray:
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
    dynamic : bool
        If True, use the static version of the function.

    Returns
    -------
    out : ndarray
          Matrix which is the result of the matrix product of the unfolding of the tensor and
          the Khatri-Rao product of the membership matrix.
    """
    if len(subs) < 3:
        log_and_raise_error(ValueError, "subs_nz should have at least 3 elements.")

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            if temporal:
                tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
            else:
                tmp *= w[0, k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)

        elif m == 2:  # we are updating v
            if temporal:
                tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
            else:
                tmp *= w[0, k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out

def log_and_raise_error(error_type, message):

    # Log the error message
    logging.error(message)

    # Raise the exception
    raise error_type(message)

#
# """
# 	Functions for handling the data.
# """
#
# import networkx as nx
# import numpy as np
# import pandas as pd
# import sktensor as skt
#
#
# def import_data(
#         dataset,
#         ego='source',
#         alter='target',
#         force_dense=True,
#         header=None,
#         sep='\\s+',
#         binary = True):
# 	"""
#     Import data, i.e. the adjacency matrix, from a given folder.
#
#     Return the NetworkX graph and its numpy adjacency matrix.
#
#     Parameters
#     ----------
#     dataset : str
#               Path of the input file.
#     ego : str
#           Name of the column to consider as source of the edge.
#     alter : str
#             Name of the column to consider as target of the edge.
#     force_dense : bool
#                   If set to True, the algorithm is forced to consider a dense adjacency tensor.
#     header : int
#              Row number to use as the column names, and the start of the data.
#
#     Returns
#     -------
#     A : list
#         List of MultiDiGraph NetworkX objects.
#     B : ndarray/sptensor
#         Graph adjacency tensor.
#     B_T : None/sptensor
#           Graph adjacency tensor (transpose).
#     data_T_vals : None/ndarray
#                   Array with values of entries A[j, i] given non-zero entry (i, j).
# 	"""
#
# 	# read adjacency file
# 	df_adj = pd.read_csv(dataset, sep=sep, header=header)
# 	print(df_adj.head())
# 	print('{0} shape: {1}'.format(dataset, df_adj.shape))
#
# 	A = read_graph(df_adj=df_adj, ego=ego, alter=alter, noselfloop=True,binary=binary)
#
# 	nodes = list(A[0].nodes())
#
# 	# save the network in a numpy tensor
# 	if force_dense:
# 		B, rw = build_B_from_A(A, nodes=nodes)
# 		B_T, data_T_vals = None, None
# 	else:
# 		B, B_T, data_T_vals, rw = build_sparse_B_from_A(A)
#
# 	print_graph_stat(A, rw)
#
# 	return A, B, B_T, data_T_vals
#
#
# def read_graph(df_adj, ego='source', alter='target', noselfloop=True,binary=True):
# 	"""
# 		Create the graph by adding edges and nodes.
# 		It assumes that columns of layers are from l+2 (included) onwards.
#
# 		Return the list MultiDiGraph NetworkX objects.
#
# 		Parameters
# 		----------
# 		df_adj : DataFrame
# 				 Pandas DataFrame object containing the edges of the graph.
# 		ego : str
# 			  Name of the column to consider as source of the edge.
# 		alter : str
# 				Name of the column to consider as target of the edge.
# 		noselfloop : bool
# 					 If set to True, the algorithm removes the self-loops.
#
# 		Returns
# 		-------
# 		A : list
# 			List of MultiDiGraph NetworkX objects.
# 	"""
#
# 	# build nodes
# 	egoID = df_adj[ego].unique()
# 	alterID = df_adj[alter].unique()
# 	nodes = list(set(egoID).union(set(alterID)))
# 	nodes.sort()
# 	L = df_adj.shape[1] - 2  # number of layers
# 	# build the NetworkX graph: create a list of graphs, as many graphs as there are layers
# 	A = [nx.MultiDiGraph() for _ in range(L)]
# 	# set the same set of nodes and order over all layers
#
# 	rec_as = np.zeros(L)
#
# 	for l in range(L):
# 		A[l].add_nodes_from(nodes)
#
# 	for index, row in df_adj.iterrows():
# 		v1 = row[ego]
# 		v2 = row[alter]
# 		for l in range(L):
# 			if row[l + 2] > 0:
# 				if A[l].has_edge(v1, v2):
# 					if binary == True:
# 						A[l][v1][v2][0]['weight'] = 1
# 					else:
# 						A[l][v1][v2][0]['weight'] += int(row[l + 2])  # the edge already exists -> no parallel edge created
# 				else:
# 					if binary == True:
# 						A[l].add_edge(v1, v2, weight=1)
# 					else:
# 						A[l].add_edge(v1, v2, weight=int(row[l + 2]))
# 	print('number edges:', A[0].number_of_edges())
#
# 	# remove self-loops
# 	if noselfloop:
# 		for l in range(L):
# 			A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))
#
# 	return A
#
#
# def print_graph_stat(A, rw):
# 	"""
# 		Print the statistics of the graph A.
#
# 		Parameters
# 		----------
# 		A : list
# 			List of MultiDiGraph NetworkX objects.
# 		rw : list
# 			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
# 	"""
#
# 	L = len(A)
# 	N = A[0].number_of_nodes()
# 	print('Number of nodes =', N)
# 	print('Number of layers =', L)
#
# 	print('Number of edges and average degree in each layer:')
# 	for l in range(L):
# 		E = A[l].number_of_edges()
# 		k = 2 * float(E) / float(N)
# 		M = np.sum([d['weight'] for u, v, d in list(A[l].edges(data=True))])
# 		kW = 2 * float(M) / float(N)
#
# 		print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')
# 		print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')
# 		print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(A[l]), 3)}')
# 		print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs) = '
# 			  f'{np.round(reciprocal_edges(A[l]), 3)}')
# 		print(f'Reciprocity (considering the weights of the edges) = {np.round(rw[l], 3)}')
#
#
# def build_B_from_A(A, nodes=None):
# 	"""
# 		Create the numpy adjacency tensor of a networkX graph.
#
# 		Parameters
# 		----------
# 		A : list
# 			List of MultiDiGraph NetworkX objects.
# 		nodes : list
# 				List of nodes IDs.
#
# 		Returns
# 		-------
# 		B : ndarray
# 			Graph adjacency tensor.
# 		rw : list
# 			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
# 	"""
#
# 	N = A[0].number_of_nodes()
# 	if nodes is None:
# 		nodes = list(A[0].nodes())
# 	B = np.empty(shape=[len(A), N, N])
# 	rw = []
# 	for l in range(len(A)):
# 		B[l, :, :] = nx.to_numpy_array(A[l], weight='weight', dtype=int, nodelist=nodes)
# 		rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())
#
# 	return B, rw
#
#
# def build_sparse_B_from_A(A):
# 	"""
# 		Create the sptensor adjacency tensor of a networkX graph.
#
# 		Parameters
# 		----------
# 		A : list
# 			List of MultiDiGraph NetworkX objects.
#
# 		Returns
# 		-------
# 		data : sptensor
# 			   Graph adjacency tensor.
# 		data_T : sptensor
# 				 Graph adjacency tensor (transpose).
# 		v_T : ndarray
# 			  Array with values of entries A[j, i] given non-zero entry (i, j).
# 		rw : list
# 			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
# 	"""
#
# 	N = A[0].number_of_nodes()
# 	L = len(A)
# 	rw = []
#
# 	d1 = np.array((), dtype='int64')
# 	d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
# 	d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
# 	v, vT, v_T = np.array(()), np.array(()), np.array(())
# 	for l in range(L):
# 		b = nx.to_scipy_sparse_matrix(A[l])
# 		b_T = nx.to_scipy_sparse_matrix(A[l]).transpose()
# 		rw.append(np.sum(b.multiply(b_T))/np.sum(b))
# 		nz = b.nonzero()
# 		nz_T = b_T.nonzero()
# 		d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
# 		d2 = np.hstack((d2, nz[0]))
# 		d2_T = np.hstack((d2_T, nz_T[0]))
# 		d3 = np.hstack((d3, nz[1]))
# 		d3_T = np.hstack((d3_T, nz_T[1]))
# 		v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
# 		vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
# 		v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
# 	subs_ = (d1, d2, d3)
# 	subs_T_ = (d1, d2_T, d3_T)
# 	data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
# 	data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)
#
# 	return data, data_T, v_T, rw
#

def reciprocal_edges(G):
	"""
		Compute the proportion of bi-directional edges, by considering the unordered pairs.

		Parameters
		----------
		G: MultiDigraph
		   MultiDiGraph NetworkX object.

		Returns
		-------
		reciprocity: float
					 Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
	"""

	n_all_edge = G.number_of_edges()
	n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
	n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network

	if n_all_edge == 0:
		raise nx.NetworkXError("Not defined for empty graphs.")

	reciprocity = float(n_overlap_edge) / float(n_undirected)

	return reciprocity
