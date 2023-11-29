"""
It is designed to compute and print statistical information about NetworkX graphs. The script
calculates metrics such as the number of nodes, layers, edges, average degree, weighted degree,
reciprocity, and more. It aims to provide a comprehensive overview of the structural properties of
the input graphs, considering both directed and weighted edges.
"""
from typing import List

import networkx as nx
import numpy as np


def print_graph_stat(A: List[nx.MultiDiGraph], rw: List[float]) -> None:
    """
    Print the statistics of the graph A.

    Parameters
    ----------
    A : list
        List of MultiDiGraph NetworkX objects.
    rw : list
         List whose elements are reciprocity (considering the weights of the edges) values,
         one per each layer.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print('Number of nodes =', N)
    print('Number of layers =', L)

    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        M = np.sum([d['weight'] for u, v, d in list(A[l].edges(data=True))])
        kW = 2 * float(M) / float(N)

        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')
        print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')
        print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(A[l]), 3)}')
        print(
            f'Reciprocity (intended as the proportion of bi-directional edges over the unordered '
            f'pairs) = {np.round(reciprocal_edges(A[l]), 3)}')
        print(
            f'Reciprocity (considering the weights of the edges) = {np.round(rw[l], 3)}'
        )


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
    n_undirected = G.to_undirected().number_of_edges(
    )  # unique pairs of edges, i.e. edges in the undirected graph
    n_overlap_edge = (
        n_all_edge - n_undirected
    )  # number of undirected edges reciprocated in the directed network

    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity
