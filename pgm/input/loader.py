"""
Functions for handling the data.
"""

from importlib.resources import files
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import networkx as nx
from numpy import ndarray
import numpy as np
import pandas as pd
from sparse import COO

from .preprocessing import build_B_from_A, build_sparse_B_from_A
from .stats import print_graph_stat, print_graph_stat_MTCOV


def import_data(
    dataset: str,
    ego: str = "source",
    alter: str = "target",
    force_dense: bool = True,
    undirected: bool = False,
    noselfloop: bool = True,
    sep: str = "\\s+",
    binary: bool = True,
    header: Optional[int] = None,
) -> tuple[
    Iterable[nx.MultiDiGraph], Union[ndarray, COO], Optional[COO], Optional[ndarray]
]:
    """
    Import data, i.e. the adjacency matrix, from a given folder.

    Return the NetworkX graph and its numpy adjacency matrix.

    Parameters
    ----------
    dataset : str
        Path of the input file.
    ego : str, optional
        Name of the column to consider as the source of the edge. Default is "source".
    alter : str, optional
        Name of the column to consider as the target of the edge. Default is "target".
    force_dense : bool, optional
        If set to True, the algorithm is forced to consider a dense adjacency tensor. Default is True.
    undirected : bool, optional
        If set to True, the algorithm considers an undirected graph. Default is False.
    noselfloop : bool, optional
        If set to True, the algorithm removes the self-loops. Default is True.
    sep : str, optional
        Separator to use when reading the dataset. Default is "\\s+".
    binary : bool, optional
        If set to True, the algorithm reads the graph with binary edges. Default is True.
    header : int, optional
        Row number to use as the column names, and the start of the data. Default is None.

    Returns
    -------
    A : list of nx.MultiDiGraph
        List of MultiDiGraph NetworkX objects representing the layers of the network.
    B : ndarray or sparse.COO
        Graph adjacency tensor. If `force_dense` is True, returns a dense ndarray. Otherwise, returns a sparse COO tensor.
    B_T : None or sparse.COO
        Transposed graph adjacency tensor. Returns None if `force_dense` is True.
    data_T_vals : None or ndarray
        Array with values of entries A[j, i] given non-zero entry (i, j). Returns None if `force_dense` is True.
    """

    # Read adjacency file
    df_adj = pd.read_csv(dataset, sep=sep, header=header)
    logging.debug(
        "Read adjacency file from %s. The shape of the data is %s.",
        dataset,
        df_adj.shape,
    )
    A = read_graph(
        df_adj=df_adj,
        ego=ego,
        alter=alter,
        undirected=undirected,
        noselfloop=noselfloop,
        binary=binary,
    )
    nodes = list(A[0].nodes())

    # Save the network in a tensor
    if force_dense:
        B, rw = build_B_from_A(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = build_sparse_B_from_A(A, calculate_reciprocity=True)

    # Get the current logging level
    current_level = logging.getLogger().getEffectiveLevel()

    # Check if the current level is INFO or lower
    if current_level <= logging.DEBUG:
        print_graph_stat(A, rw)

    return A, B, B_T, data_T_vals


def import_data_mtcov(
    in_folder: str,
    adj_name: str = "adj.csv",
    cov_name: str = "X.csv",
    ego: str = "source",
    egoX: str = "Name",
    alter: str = "target",
    attr_name: str = "Metadata",
    undirected: bool = False,
    force_dense: bool = True,
    noselfloop: bool = True,
) -> Tuple[
    List[nx.MultiDiGraph], Union[ndarray, COO], Optional[pd.DataFrame], List[str]
]:
    """
    Import data, i.e. the adjacency tensor and the design matrix, from a given folder.

    Parameters
    ----------
    in_folder : str
        Path of the folder containing the input files.
    adj_name : str, optional
        Input file name of the adjacency tensor. Default is "adj.csv".
    cov_name : str, optional
        Input file name of the design matrix. Default is "X.csv".
    ego : str, optional
        Name of the column to consider as the source of the edge. Default is "source".
    egoX : str, optional
        Name of the column to consider as node IDs in the design matrix-attribute dataset. Default is "Name".
    alter : str, optional
        Name of the column to consider as the target of the edge. Default is "target".
    attr_name : str, optional
        Name of the attribute to consider in the analysis. Default is "Metadata".
    undirected : bool, optional
        If set to True, the algorithm considers an undirected graph. Default is False.
    force_dense : bool, optional
        If set to True, the algorithm is forced to consider a dense adjacency tensor. Default is True.
    noselfloop : bool, optional
        If set to True, the algorithm removes the self-loops. Default is True.

    Returns
    -------
    A : list of nx.MultiDiGraph
        List of MultiDiGraph NetworkX objects representing the layers of the network.
    B : ndarray or sparse.COO
        Graph adjacency tensor. If `force_dense` is True, returns a dense ndarray. Otherwise, returns a sparse COO tensor.
    X_attr : pd.DataFrame or None
        Pandas DataFrame object representing the one-hot encoding version of the design matrix. Returns None if the design matrix is not provided.
    nodes : list of str
        List of node IDs.
    """

    def get_data_path(in_folder):
        """
        Try to treat in_folder as a package data path, if that fails, treat in_folder as a file path.
        """
        try:
            # Try to treat in_folder as a package data path
            return files(in_folder)
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            # If that fails, treat in_folder as a file path
            return Path(in_folder)

    # Check if in_folder is a package data path or a file path
    in_folder_path = get_data_path(in_folder)

    # Read the adjacency file
    logging.debug("Reading adjacency file...")
    df_adj = pd.read_csv(in_folder_path / adj_name)  # read adjacency file
    logging.debug("Adjacency shape: %s", df_adj.shape)

    df_X = pd.read_csv(
        in_folder_path / cov_name
    )  # read the csv file with the covariates
    logging.debug("Indiv shape: %s", df_X.shape)

    # Create the graph adding nodes and edges
    A = read_graph(
        df_adj=df_adj,
        ego=ego,
        alter=alter,
        undirected=undirected,
        noselfloop=noselfloop,
        binary=False,
    )

    nodes = list(A[0].nodes)

    # Get the current logging level
    current_level = logging.getLogger().getEffectiveLevel()

    # Check if the current level is INFO or lower
    if current_level <= logging.DEBUG:
        print_graph_stat_MTCOV(A)

    # Save the multilayer network in a tensor with all layers
    if force_dense:
        B, _ = build_B_from_A(A, nodes=nodes, calculate_reciprocity=False)
    else:
        B = build_sparse_B_from_A(A)

    # Read the design matrix with covariates
    X_attr = read_design_matrix(df_X, nodes, attribute=attr_name, ego=egoX)

    return A, B, X_attr, nodes


def read_graph(
    df_adj: pd.DataFrame,
    ego: str = "source",
    alter: str = "target",
    undirected: bool = False,
    noselfloop: bool = True,
    binary: bool = True,
    label: str = "weight",
) -> List:
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
    binary: bool
            If set to True, read the graph with binary edges.
    label: str
             Name of the column to consider as the edge weight.

    Returns
    -------
    A: list
       List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # Build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = sorted(set(egoID).union(set(alterID)))

    L = df_adj.shape[1] - 2  # number of layers
    # Build the multilayer NetworkX graph: create a list of graphs, as many
    # graphs as there are layers
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]

    logging.debug("Creating the network ...")
    # Set the same set of nodes and order over all layers
    for layer in range(L):
        A[layer].add_nodes_from(nodes)

    for _, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for layer in range(L):
            if row[layer + 2] > 0:
                if binary:
                    if A[layer].has_edge(v1, v2):
                        A[layer][v1][v2][0][label] = 1
                    else:
                        edge_attributes = {label: 1}
                        A[layer].add_edge(v1, v2, **edge_attributes)
                else:
                    if A[layer].has_edge(v1, v2):
                        A[layer][v1][v2][0][label] += int(
                            row[layer + 2]
                        )  # the edge already exists, no parallel edge created
                    else:
                        edge_attributes = {label: int(row[layer + 2])}
                        A[layer].add_edge(v1, v2, **edge_attributes)

    # Remove self-loops
    if noselfloop:
        logging.debug("Removing self loops")
        for layer in range(L):
            A[layer].remove_edges_from(list(nx.selfloop_edges(A[layer])))

    return A


def read_design_matrix(
    df_X: pd.DataFrame,
    nodes: List,
    attribute: Union[str, None] = None,
    ego: str = "Name",
):
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

    Returns
    -------
    X_attr : DataFrame
             Pandas DataFrame that represents the one-hot encoding version of the design matrix.
    """
    logging.debug("Reading the design matrix...")

    X = df_X[df_X[ego].isin(nodes)]  # filter nodes
    X = X.set_index(ego).loc[nodes].reset_index()  # sort by nodes

    if attribute is None:
        X_attr = pd.get_dummies(X.iloc[:, 1])  # gets the first columns after the ego
    else:  # use one attribute as it is
        X_attr = pd.get_dummies(X[attribute])

    logging.debug("Design matrix shape: %s", X_attr.shape)
    logging.debug("Distribution of attribute %s:", attribute)
    logging.debug("%s", np.sum(X_attr, axis=0))

    return X_attr
