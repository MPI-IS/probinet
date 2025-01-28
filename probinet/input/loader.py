"""
Functions for handling the data.
"""

import csv
import logging
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from ..models.classes import GraphData
from ..utils.tools import log_and_raise_error
from .preprocessing import (
    create_adjacency_tensor_from_graph_list,
    create_sparse_adjacency_tensor_from_graph_list,
)
from .stats import print_graph_stats


def build_adjacency_from_networkx(
    network: nx.Graph,
    weight_list: list[str],
    file_name: Optional[PathLike] = None,
) -> GraphData:
    """
    Import networkx graph and convert it to the GraphData object

    Parameters
    ----------
    networkx
        networkx graph that will be converted to GraphData object
    weight_list
        list of names of weights user would like to use from networkx graph
    file_name
        name of csv file (and path) created from networkx graph (used to create GraphData object)
        e.g. /path/to/file/file_name.csv
    Returns
    -------
    GraphData
        GraphData object created from networkx graph
    """
    attribute_names = {key for _, _, data in network.edges(data=True) for key in data}
    for w in weight_list:
        assert w in attribute_names, f"{w} is not an attribute"

    if not file_name or Path(file_name).suffix != ".csv":
        file_name = Path.cwd() / "edge_list.csv"
        logging.DEBUG("File will be stored at %s" % file_name)

    # Save edges to a CSV file
    with open(file_name, "w", newline="", encoding="utf-8") as edge_file:
        writer = csv.writer(edge_file, delimiter=" ")
        # Write header
        writer.writerow(["source", "target"] + weight_list)  # Get edge keys.
        # Write edge data
        for source, target, attrs in network.edges(data=True):
            writer.writerow([source, target] + [attrs[a] for a in weight_list])

    return build_adjacency_from_file(file_name)


def build_adjacency_from_file(
    path_to_file: PathLike,
    ego: str = "source",
    alter: str = "target",
    force_dense: bool = True,
    undirected: bool = False,
    noselfloop: bool = True,
    sep: str = "\\s+",
    binary: bool = True,
    header: Optional[int] = 0,
    **_kwargs: Any,
) -> GraphData:
    """
    Import data, i.e., the adjacency matrix, from a given folder.

    Return the NetworkX graph and its numpy adjacency matrix.

    Parameters
    ----------
    path_to_file
        Path of the input file.
    ego
        Name of the column to consider as the source of the edge.
    alter
        Name of the column to consider as the target of the edge.
    force_dense
        If set to True, the algorithm is forced to consider a dense adjacency tensor.
    undirected
        If set to True, the algorithm considers an undirected graph.
    noselfloop
        If set to True, the algorithm removes the self-loops.
    sep
        Separator to use when reading the dataset.
    binary
        If set to True, the algorithm reads the graph with binary edges.
    header
        Row number to use as the column names, and the start of the data.

    Returns
    -------
    GraphData
        Named tuple containing the graph list, the adjacency tensor, the transposed tensor,
        the data values, and the nodes.
    """

    # Read adjacency file
    df_adj = pd.read_csv(path_to_file, sep=sep, header=header)
    logging.debug(
        "Read adjacency file from %s. The shape of the data is %s.",
        path_to_file,
        df_adj.shape,
    )
    # Check that the df has only non negative values; if not, raise an error
    if (df_adj.iloc[:, 2:] < 0).any(axis=None):
        # We check this for the columns that contain weights (i.e., from the 2nd column onwards)
        log_and_raise_error(ValueError, "There are negative weights.")

    # Build a list of MultiDiGraph NetworkX objects representing the layers of the network
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
        B, rw = create_adjacency_tensor_from_graph_list(A, nodes=nodes)
        B_T, data_T_vals = None, None
    else:
        B, B_T, data_T_vals, rw = create_sparse_adjacency_tensor_from_graph_list(
            A, calculate_reciprocity=True
        )

    # Get the current logging level
    current_level = logging.getLogger().getEffectiveLevel()

    # Check if the current level is INFO or lower
    if current_level <= logging.DEBUG:
        print_graph_stats(A, rw)

    return GraphData(
        graph_list=A,
        adjacency_tensor=B,
        transposed_tensor=B_T,
        data_values=data_T_vals,
        nodes=nodes,
    )


def read_and_process_design_matrix(
    in_folder_path: PathLike,
    cov_name: str,
    sep: str,
    header: Optional[int],
    nodes: list[str],
    attr_name: str,
    egoX: str,
) -> pd.DataFrame:
    """
    Read and process the design matrix with covariates.

    Parameters
    ----------
    in_folder_path
        Path to the folder containing the input files.
    cov_name
        Name of the covariate file.
    sep : str
        Separator to use when reading the covariate file.
    header
        Row number to use as the column names, and the start of the data.
    nodes
        List of node IDs.
    attr_name
        Name of the attribute to consider in the analysis.
    egoX : str
        Name of the column to consider as node IDs in the design matrix.

    Returns
    -------
    X_attr
        Pandas DataFrame that represents the one-hot encoding version of the design matrix.
    """
    df_X = pd.read_csv(in_folder_path / cov_name, sep=sep, header=header)
    logging.debug("Indiv shape: %s", df_X.shape)

    # Read and return the design matrix with covariates
    return read_design_matrix(df_X, nodes, attribute=attr_name, ego=egoX)


def build_adjacency_and_design_from_file(
    in_folder: str,
    adj_name: str = "multilayer_network.csv",
    cov_name: str = "X.csv",
    ego: str = "source",
    egoX: str = "Name",
    alter: str = "target",
    attr_name: str = "Metadata",
    undirected: bool = False,
    force_dense: bool = True,
    noselfloop: bool = True,
    sep: str = ",",
    header: Optional[int] = 0,
    return_X_as_np: bool = True,
    **_kwargs,
) -> GraphData:
    """
    Import data, i.e. the adjacency tensor and the design matrix, from a given folder.

    Parameters
    ----------
    in_folder : str
        Path of the folder containing the input files.
    adj_name : str
        Input file name of the adjacency tensor.
    cov_name : str
        Input file name of the design matrix.
    ego : str
        Name of the column to consider as the source of the edge.
    egoX : str
        Name of the column to consider as node IDs in the design matrix-attribute dataset.
    alter : str
        Name of the column to consider as the target of the edge.
    attr_name : str
        Name of the attribute to consider in the analysis.
    undirected : bool
        If set to True, the algorithm considers an undirected graph.
    force_dense : bool
        If set to True, the algorithm is forced to consider a dense adjacency tensor.
    noselfloop : bool
        If set to True, the algorithm removes the self-loops.
    sep : str
        Separator to use when reading the dataset.
    header : int
        Row number to use as the column names, and the start of the data.
    return_X_as_np : bool
        If set to True, the design matrix is returned as a numpy array.
    _kwargs
        Additional keyword arguments.

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
        The case where the input is a file path refers to the case where the user points to data
        outside the package.
        """
        try:
            # Try to treat in_folder as a package data path
            return files(in_folder)
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            # If that fails, treat in_folder as a file path
            return Path(in_folder)

    # Check if in_folder is a package data path or a file path
    in_folder_path = get_data_path(in_folder)

    # Build the adjacency tensor and the incidence tensor
    A, B, _, _, nodes, _ = build_adjacency_from_file(
        path_to_file=in_folder_path / adj_name,
        ego=ego,
        alter=alter,
        force_dense=force_dense,
        undirected=undirected,
        noselfloop=noselfloop,
        sep=sep,
        binary=False,
        header=header,
    )

    # Read the design matrix with covariates
    X_df = read_and_process_design_matrix(
        in_folder_path, cov_name, sep, header, nodes, attr_name, egoX
    )

    if return_X_as_np:
        # Convert X_df to a numpy array
        X_df = np.array(X_df)

    return GraphData(graph_list=A, adjacency_tensor=B, design_matrix=X_df, nodes=nodes)


def read_graph(
    df_adj: pd.DataFrame,
    ego: str = "source",
    alter: str = "target",
    undirected: bool = False,
    noselfloop: bool = True,
    binary: bool = True,
    label: str = "weight",
) -> list[nx.MultiDiGraph]:
    """
    Create the graph by adding edges and nodes.

    Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects. The graph
    is built by adding edges and nodes from the given DataFrame. The graphs listed in the output
    have an edge attribute named `label`.

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
             Name to be assigned to the edge attribute, across all layers.

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

    logging.debug("Creating the networks ...")
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
    nodes: list,
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
