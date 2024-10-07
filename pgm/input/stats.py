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

    logging.info("Number of nodes = %s", N)
    logging.info("Number of layers = %s", L)

    logging.info("Number of edges and average degree in each layer:")
    for layer in range(L):
        E = G[layer].number_of_edges()
        k = 2 * float(E) / float(N)
        logging.info("E[%s] = %s - <k> = %.2f", layer, E, k)
        weights = [d["weight"] for u, v, d in list(G[layer].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            M = np.sum([d["weight"] for u, v, d in list(G[layer].edges(data=True))])
            kW = 2 * float(M) / float(N)
            logging.info("M[%s] = %s - <k_weighted> = %.3f", layer, M, kW)

        logging.info("Sparsity [%s] = %.3f", layer, E / (N * N))
        logging.info("Reciprocity (networkX) = %.3f", nx.reciprocity(G[layer]))

        if rw is not None:
            logging.info(
                "Reciprocity (considering the weights of the edges) = %.3f",
                rw[layer],
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
    for layer in range(L):
        E = A[layer].number_of_edges()
        k = 2 * float(E) / float(N)
        avg_edges += E
        avg_density += k
        logging.info("E[%s] = %s - <k> = %.3f", layer, E, k)

        weights = [d["weight"] for u, v, d in list(A[layer].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            unweighted = False
            M = np.sum([d["weight"] for u, v, d in list(A[layer].edges(data=True))])
            kW = 2 * float(M) / float(N)
            avg_M += M
            avg_densityW += kW
            logging.info("M[%s] = %s - <k_weighted> = %.3f", layer, M, kW)

        logging.info("Sparsity [%s] = %.3f", layer, E / (N * N))

    logging.info("\nAverage edges over all layers: %.3f", avg_edges / L)
    logging.info("Average degree over all layers: %.2f", avg_density / L)
    logging.info("Total number of edges: %s", avg_edges)
    if not unweighted:
        logging.info("Average edges over all layers (weighted): %.3f", avg_M / L)
        logging.info(
            "Average degree over all layers (weighted): %.2f", avg_densityW / L
        )
        logging.info("Total number of edges (weighted): %.3f", avg_M)
    logging.info("Sparsity = %.3f", avg_edges / (N * N * L))


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


def probabilities(
    structure: str,
    sizes: List[int],
    N: int = 100,
    K: int = 2,
    avg_degree: float = 4.0,
    alpha: float = 0.1,
    beta: Optional[float] = None,
) -> np.ndarray:
    """
    Return the CxC array with probabilities between and within groups.

    Parameters
    ----------
    structure : str
                Structure of the layer, e.g. assortative, disassortative, core-periphery or directed-biased.
    sizes : List[int]
            List with the sizes of blocks.
    N : int
        Number of nodes.
    K : int
        Number of communities.
    avg_degree : float
                 Average degree over the nodes.
    alpha : float
            Alpha value. Default is 0.1.
    beta : float
           Beta value. Default is 0.3 * alpha.

    Returns
    -------
    p : np.ndarray
        Array with probabilities between and within groups.
    """

    if beta is None:
        beta = alpha * 0.3
    p1 = avg_degree * K / N
    if structure == "assortative":
        p = p1 * alpha * np.ones((len(sizes), len(sizes)))  # secondary-probabilities
        np.fill_diagonal(p, p1)  # primary-probabilities
    elif structure == "disassortative":
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(p, alpha * p1)
    elif structure == "core-periphery":
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(np.fliplr(p), alpha * p1)
        p[1, 1] = beta * p1
    elif structure == "directed-biased":
        p = alpha * p1 * np.ones((len(sizes), len(sizes)))
        p[0, 1] = p1
        p[1, 0] = beta * p1

    return p
