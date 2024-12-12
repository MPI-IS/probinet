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

from ..utils.tools import log_and_raise_error


def print_graph_stats(
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

    print("Number of nodes = %s" % N)
    print("Number of layers = %s" % L)

    print("Number of edges and average degree in each layer:")
    for layer in range(L):
        E = G[layer].number_of_edges()
        k = 2 * float(E) / float(N)
        print("E[%s] = %s / <k> = %.2f" % (layer, E, k))
        weights = [d["weight"] for u, v, d in list(G[layer].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            M = np.sum([d["weight"] for u, v, d in list(G[layer].edges(data=True))])
            kW = 2 * float(M) / float(N)
            print("M[%s] = %s - <k_weighted> = %.3f" % (layer, M, kW))

        print("Sparsity [%s] = %.3f" % (layer, E / (N * N)))
        # Check if network is directed; if not, skip the calculation of the reciprocity
        if nx.is_directed(G[layer]):
            print("Reciprocity (networkX) = %.3f" % nx.reciprocity(nx.Graph(G[layer])))

        if rw is not None:
            print(
                "Reciprocity (considering the weights of the edges) = %.3f" % rw[layer]
            )


def print_graph_stats_MTCOV(A: List[nx.MultiDiGraph]) -> None:
    """
    Print the statistics of the graph A.

    Parameters
    ----------
    A : list
        List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print("Number of edges and average degree in each layer:")
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
        print("E[%s] = %s - <k> = %.3f" % (layer, E, k))

        weights = [d["weight"] for u, v, d in list(A[layer].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            unweighted = False
            M = np.sum([d["weight"] for u, v, d in list(A[layer].edges(data=True))])
            kW = 2 * float(M) / float(N)
            avg_M += M
            avg_densityW += kW
            print("M[%s] = %s - <k_weighted> = %.3f" % (layer, M, kW))

        print("Sparsity [%s] = %.3f" % (layer, E / (N * N)))

    print("\nAverage edges over all layers: %.3f" % (avg_edges / L))
    print("Average degree over all layers: %.2f" % (avg_density / L))
    print("Total number of edges: %s" % avg_edges)
    if not unweighted:
        print("Average edges over all layers (weighted): %.3f" % (avg_M / L))
        print("Average degree over all layers (weighted): %.2f" % (avg_densityW / L))
        print("Total number of edges (weighted): %.3f" % avg_M)
    print("Sparsity = %.3f" % (avg_edges / (N * N * L)))


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
