"""
It is designed to compute and print statistical information about NetworkX graphs. The script
calculates metrics such as the number of nodes, layers, edges, average degree, weighted degree,
reciprocity, and more. It aims to provide a comprehensive overview of the structural properties of
the input graphs, considering both directed and weighted edges.
"""

import logging
from typing import List, Optional

import networkx as nx
import numpy as np

from .tools import log_and_raise_error


def print_graph_stat(
    G: List[nx.MultiDiGraph], rw: Optional[List[float]] = None
) -> None:
    """
    Print the statistics of the graph G.

    This function calculates and prints various statistics of the input graph such as the number of edges,
    average degree in each layer, sparsity, and reciprocity. If the weights of the edges are provided,
    it also calculates and prints the reciprocity considering the weights of the edges.

    Parameters
    ----------
    G : list
        List of MultiDiGraph NetworkX objects representing the layers of the graph.
    rw : list, optional
        List of floats representing the weights of the edges in each layer of the graph.
        If not provided, the function will consider the graph as unweighted.

    """

    L = len(G)
    N = G[0].number_of_nodes()

    logging.info("Number of edges and average degree in each layer:")
    for l in range(L):
        E = G[l].number_of_edges()
        k = 2 * float(E) / float(N)
        logging.info("E[%s] = %s - <k> = %s", l, E, np.round(k, 3))

        weights = [d["weight"] for u, v, d in list(G[l].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            M = np.sum([d["weight"] for u, v, d in list(G[l].edges(data=True))])
            kW = 2 * float(M) / float(N)
            logging.info("M[%s] = %s - <k_weighted> = %s", l, M, np.round(kW, 3))

        logging.info("Sparsity [%s] = %s", l, np.round(E / (N * N), 3))

        logging.info("Reciprocity (networkX) = %s", np.round(nx.reciprocity(G[l]), 3))

        if rw is not None:
            logging.info(
                "Reciprocity (considering the weights of the edges) = %s",
                np.round(rw[l], 3),
            )


def print_graph_stat_MTCOV(A: List[nx.MultiDiGraph]) -> None:
    """
    Print the statistics of the graph A.

    Parameters
    ----------
    A : list
        List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    logging.info("Number of edges and average degree in each layer:")
    avg_edges = 0.0
    avg_density = 0.0
    avg_M = 0.0
    avg_densityW = 0.0
    unweighted = True
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        avg_edges += E
        avg_density += k
        logging.info("E[%s] = %s - <k> = %s", l, E, np.round(k, 3))

        weights = [d["weight"] for u, v, d in list(A[l].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            unweighted = False
            M = np.sum([d["weight"] for u, v, d in list(A[l].edges(data=True))])
            kW = 2 * float(M) / float(N)
            avg_M += M
            avg_densityW += kW
            logging.info("M[%s] = %s - <k_weighted> = %s", l, M, np.round(kW, 3))

        logging.info("Sparsity [%s] = %s", l, np.round(E / (N * N), 3))

    logging.info("\nAverage edges over all layers: %s", np.round(avg_edges / L, 3))
    logging.info("Average degree over all layers: %s", np.round(avg_density / L, 2))
    logging.info("Total number of edges: %s", avg_edges)
    if not unweighted:
        logging.info(
            "Average edges over all layers (weighted):", np.round(avg_M / L, 3)
        )
        logging.info(
            "Average degree over all layers (weighted):", np.round(avg_densityW / L, 2)
        )
        logging.info("Total number of edges (weighted):", avg_M)
    logging.info("Sparsity = %s", np.round(avg_edges / (N * N * L), 3))


def reciprocal_edges(G: nx.MultiDiGraph) -> float:
    """
    Compute the proportion of bi-directional edges, by considering the unordered pairs.

    Parameters
    ----------
    G: MultiDigraph
       MultiDiGraph NetworkX object.

    Returns
    -------
    reciprocity: float
                 Reciprocity value, intended as the proportion of bi-directional edges over the
                 unordered pairs.
    """

    n_all_edge = G.number_of_edges()
    # unique pairs of edges, i.e. edges in the undirected graph
    n_undirected = G.to_undirected().number_of_edges()
    # number of undirected edges reciprocated in the directed network
    n_overlap_edge = n_all_edge - n_undirected

    if n_all_edge == 0:
        log_and_raise_error(nx.NetworkXError, "Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity
