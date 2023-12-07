"""
Functions for handling the data.
"""
from typing import Any, Iterable, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from numpy import ndarray

from . import preprocessing as prep
from .statistics import print_graph_stat


# TODO: Correct the docstring, the type hints and the return type.
def import_data(dataset: str,
                ego: str = 'source',
                alter: str = 'target',
                force_dense: bool = True,
                undirected=False,
                noselfloop=True,
                verbose=True,
                binary=True,
                header: Optional[int] = None) -> Tuple[
    Iterable, Union[ndarray, Any], Optional[Any], Optional[ndarray]]:
    """
    Import data, i.e. the adjacency matrix, from a given folder.

    Return the NetworkX graph and its numpy adjacency matrix.

    Parameters
    ----------
    dataset : str
              Path of the input file.
    ego : str
          Name of the column to consider as the source of the edge.
    alter : str
            Name of the column to consider as the target of the edge.
    force_dense : bool
                  If set to True, the algorithm is forced to consider a dense adjacency tensor.
    header : int
             Row number to use as the column names, and the start of the data.

    Returns
    -------
    A : list
        List of MultiDiGraph NetworkX objects.
    B : ndarray/sptensor
        Graph adjacency tensor.
    B_T : None/sptensor
          Graph adjacency tensor (transpose).
    data_T_vals : None/ndarray
                  Array with values of entries A[j, i] given non-zero entry (i, j).
    """

    # read adjacency file
    df_adj = pd.read_csv(dataset, sep='\\s+', header=header)
    print(f"{dataset} shape: {df_adj.shape}")

    # A = read_graph(df_adj=df_adj, ego=ego, alter=alter, noselfloop=True)
    A = read_graph(df_adj=df_adj, ego=ego, alter=alter, undirected=undirected, noselfloop=noselfloop,
                   verbose=verbose,
                   binary=binary)
    nodes = list(A[0].nodes())
    print('\nNumber of nodes =', len(nodes))
    print('Number of layers =', len(A))
    # save the network in a tensor
    if force_dense:
        B, rw = prep.build_B_from_A(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = prep.build_sparse_B_from_A(A)
    if verbose:
        print_graph_stat(A, rw)

    return A, B, B_T, data_T_vals


def read_graph(df_adj, ego='source', alter='target', undirected=False, noselfloop=True, verbose=True, binary=True):
    """
        Create the graph by adding edges and nodes.

        Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

        Parameters
        ----------
        df_adj : DataFrame
                 Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.
        verbose : bool
                  Flag to print details.
        binary : bool
                 If set to True, read the graph with binary edges.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]

    if verbose:
        print('Creating the network ...', end=' ')
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2] > 0:
                if binary:
                    if A[l].has_edge(v1, v2):
                        A[l][v1][v2][0]['weight'] = 1
                    else:
                        A[l].add_edge(v1, v2, weight=1)
                else:
                    if A[l].has_edge(v1, v2):
                        A[l][v1][v2][0]['weight'] += int(
                            row[l + 2])  # the edge already exists, no parallel edge created
                    else:
                        A[l].add_edge(v1, v2, weight=int(row[l + 2]))
    if verbose:
        print('done!')

    # remove self-loops
    if noselfloop:
        if verbose:
            print('Removing self loops')
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A
