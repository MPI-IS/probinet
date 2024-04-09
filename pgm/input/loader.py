"""
Functions for handling the data.
"""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import networkx as nx
from numpy import ndarray
import numpy as np
import pandas as pd
from sktensor import sptensor

from .preprocessing import build_B_from_A, build_sparse_B_from_A
from .stats import print_graph_stat


def import_data(dataset: Path,
                ego: str = 'source',
                alter: str = 'target',
                force_dense: bool = True,
                undirected=False,
                noselfloop=True,
                verbose=True,
                binary=True,
                header: Optional[int] = None) -> Tuple[
        List, Union[ndarray, Any], Optional[Any], Optional[ndarray]]:
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
    if verbose:  # TODO: Add a logger
        print(f"{dataset} shape: {df_adj.shape}")
    if verbose:
        print(f"{dataset} shape: {df_adj.shape}")

    # A = read_graph(df_adj=df_adj, ego=ego, alter=alter, noselfloop=True)
    A = read_graph(
        df_adj=df_adj,
        ego=ego,
        alter=alter,
        undirected=undirected,
        noselfloop=noselfloop,
        verbose=verbose,
        binary=binary)
    nodes = list(A[0].nodes())
    if verbose:
        print('\nNumber of nodes =', len(nodes))
        print('Number of layers =', len(A))
    # save the network in a tensor
    if force_dense:
        B, rw = build_B_from_A(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = build_sparse_B_from_A(A, calculate_reciprocity=True)
    if verbose:
        print_graph_stat(A, rw)

    return A, B, B_T, data_T_vals


def import_data_mtcov(
        in_folder: Union[str, Path],
        adj_name: Union[str, Path] = 'adj.csv',
        cov_name: Union[str, Path] = 'X.csv',
        ego: str = 'source',
        egoX: str = 'Name',
        alter: str = 'target',
        attr_name: str = 'Metadata',
        undirected: bool = False,
        force_dense: bool = True,
        noselfloop: bool = True,
        verbose: bool = True) -> Tuple[List, Union[sptensor, Any], Optional[Any], List]:
    """
    Import data, i.e. the adjacency tensor and the design matrix, from a given folder.

    Return the NetworkX graph, its numpy adjacency tensor and the dummy version of the design matrix.

    Parameters
    ----------
    in_folder : str
                Path of the folder containing the input files.
    adj_name : str
               Input file name of the adjacency tensor.
    cov_name : str
               Input file name of the design matrix.
    ego : str
          Name of the column to consider as source of the edge.
    egoX : str
           Name of the column to consider as node IDs in the design matrix-attribute dataset.
    alter : str
            Name of the column to consider as target of the edge.
    attr_name : str
                Name of the attribute to consider in the analysis.
    undirected : bool
                 If set to True, the algorithm considers an undirected graph.
    force_dense : bool
                  If set to True, the algorithm is forced to consider a dense adjacency tensor.
    noselfloop : bool
                 If set to True, the algorithm removes the self-loops.
    verbose : bool
              Flag to print details.

    Returns
    -------
    A : list
        List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    B : ndarray/sptensor
        Graph adjacency tensor.
    X_attr : DataFrame
             Pandas DataFrame object representing the one-hot encoding version of the design matrix.
    nodes : list
            List of nodes IDs.
    """

    # df_adj = pd.read_csv(in_folder + adj_name, index_col=0) # read adjacency file
    df_adj = pd.read_csv(in_folder/ adj_name)  # read adjacency file
    print('\nAdjacency shape: {0}'.format(df_adj.shape))

    df_X = pd.read_csv(in_folder / cov_name)  # read the csv file with the covariates
    print('Indiv shape: ', df_X.shape)

    # create the graph adding nodes and edges
    A = read_graph(
        df_adj=df_adj,
        ego=ego,
        alter=alter,
        undirected=undirected,
        noselfloop=noselfloop,
        verbose=verbose,
        binary=False)

    nodes = list(A[0].nodes)
    print('\nNumber of nodes =', len(nodes))
    print('Number of layers =', len(A))
    if verbose:
        print_graph_stat(A)

    # save the multilayer network in a tensor with all layers
    if force_dense:
        B = build_B_from_A(A, nodes=nodes, calculate_reciprocity=False)
    else:
        B = build_sparse_B_from_A(A)

    # read the design matrix with covariates
    X_attr = read_design_matrix(df_X, nodes, attribute=attr_name, ego=egoX, verbose=verbose)

    return A, B, X_attr, nodes


def read_graph(
        df_adj: pd.DataFrame,
        ego: str = 'source',
        alter: str = 'target',
        undirected: bool = False,
        noselfloop: bool = True,
        verbose: bool = True,
        binary: bool = True) -> List:
    """
    Create the graph by adding edges and nodes.

    Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

    Parameters
    ----------
    df_adj: DataFrame
             Pandas DataFrame object containing the edges of the graph.
    ego: str
          Name of the column to consider as the source of the edge.
    alter: str
            Name of the column to consider as the target of the edge.
    undirected: bool
                 If set to True, the algorithm considers an undirected graph.
    noselfloop: bool
                 If set to True, the algorithm removes the self-loops.
    verbose: bool
              Flag to print details.
    binary: bool
             If set to True, read the graph with binary edges.

    Returns
    -------
    A: list
        List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the multilayer NetworkX graph: create a list of graphs, as many
    # graphs as there are layers
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]

    if verbose:
        print('Creating the network ...', end=' ')
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for _, row in df_adj.iterrows():
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

    # remove self-loops
    if noselfloop:
        if verbose:
            print('Removing self loops')
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A


def read_design_matrix(df_X: pd.DataFrame,
                       nodes: List,
                       attribute: Union[str, None] = None,
                       ego: str = 'Name',
                       verbose: bool = True):
    """
        Create the design matrix with the one-hot encoding of the given attribute.

        Parameters
        ----------
        df_X : DataFrame
               Pandas DataFrame object containing the covariates of the nodes.
        nodes : list
                List of nodes IDs.
        attribute : str
                    Name of the attribute to consider in the analysis.
        ego : str
              Name of the column to consider as node IDs in the design matrix.
        verbose : bool
                  Flag to print details.

        Returns
        -------
        X_attr : DataFrame
                 Pandas DataFrame that represents the one-hot encoding version of the design matrix.
    """

    X = df_X[df_X[ego].isin(nodes)]  # filter nodes
    X = X.set_index(ego).loc[nodes].reset_index()  # sort by nodes

    if attribute is None:
        X_attr = pd.get_dummies(X.iloc[:, 1])  # gets the first columns after the ego
    else:  # use one attribute as it is
        X_attr = pd.get_dummies(X[attribute])
    print('\nDesign matrix shape: ', X_attr.shape)

    if verbose:
        print('Distribution of attribute {0}: '.format(attribute))
        print(np.sum(X_attr, axis=0))

    return X_attr
