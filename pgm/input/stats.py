"""
It is designed to compute and print statistical information about NetworkX graphs. The script
calculates metrics such as the number of nodes, layers, edges, average degree, weighted degree,
reciprocity, and more. It aims to provide a comprehensive overview of the structural properties of
the input graphs, considering both directed and weighted edges.
"""
from typing import List, Optional

import networkx as nx
import numpy as np


# pylint: disable=too-many-arguments, too-many-instance-attributes,
# too-many-locals, too-many-branches, too-many-statements
def print_graph_stat(G: List[nx.MultiDiGraph], rw: Optional[List[float]] = None) -> None:
    """
    Print the statistics of the graph A.

    Parameters
    ----------
    G : list
        List of MultiDiGraph NetworkX objects.
    """

    L = len(G)
    N = G[0].number_of_nodes()

    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = G[l].number_of_edges()
        k = 2 * float(E) / float(N)
        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')

        weights = [d['weight'] for u, v, d in list(G[l].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            M = np.sum([d['weight'] for u, v, d in list(G[l].edges(data=True))])
            kW = 2 * float(M) / float(N)
            print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')

        print(f'Sparsity [{l}] = {np.round(E / (N * N), 3)}')

        print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(G[l]), 3)}')
        print(
            f'Reciprocity (intended as the proportion of bi-directional edges over the unordered '
            f'pairs) = {np.round(reciprocal_edges(G[l]), 3)}\n')

        if rw is not None:
            print(
                f'Reciprocity (considering the weights of the edges) = {np.round(rw[l], 3)}'
            )


def print_graph_stat_MTCov(A: List[nx.MultiDiGraph]) -> None:
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print('Number of edges and average degree in each layer:')
    avg_edges = 0.
    avg_density = 0.
    avg_M = 0.
    avg_densityW = 0.
    unweighted = True
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        avg_edges += E
        avg_density += k
        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')

        weights = [d['weight'] for u, v, d in list(A[l].edges(data=True))]
        if not np.array_equal(weights, np.ones_like(weights)):
            unweighted = False
            M = np.sum([d['weight'] for u, v, d in list(A[l].edges(data=True))])
            kW = 2 * float(M) / float(N)
            avg_M += M
            avg_densityW += kW
            print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')

        print(f'Sparsity [{l}] = {np.round(E / (N * N), 3)}')

    print('\nAverage edges over all layers:', np.round(avg_edges / L, 3))
    print('Average degree over all layers:', np.round(avg_density / L, 2))
    print('Total number of edges:', avg_edges)
    if not unweighted:
        print('Average edges over all layers (weighted):', np.round(avg_M / L, 3))
        print('Average degree over all layers (weighted):', np.round(avg_densityW / L, 2))
        print('Total number of edges (weighted):', avg_M)
    print(f'Sparsity = {np.round(avg_edges / (N * N * L), 3)}')


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
        raise nx.NetworkXError("Not defined for empty graphs.")

    reciprocity = float(n_overlap_edge) / float(n_undirected)

    return reciprocity


def probabilities(
        structure: str,
        sizes: List[int],
        N: int = 100,
        C: int = 2,
        avg_degree: float = 4.) -> np.ndarray:
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
    C : int
        Number of communities.
    avg_degree : float
                 Average degree over the nodes.

    Returns
    -------
    p : np.ndarray
        Ar
    """
    alpha = 0.1
    beta = alpha * 0.3
    p1 = avg_degree * C / N
    if structure == 'assortative':
        p = p1 * alpha * np.ones((len(sizes), len(sizes)))  # secondary-probabilities
        np.fill_diagonal(p, p1)  # primary-probabilities
    elif structure == 'disassortative':
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(p, alpha * p1)
    elif structure == 'core-periphery':
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(np.fliplr(p), alpha * p1)
        p[1, 1] = beta * p1
    elif structure == 'directed-biased':
        p = alpha * p1 * np.ones((len(sizes), len(sizes)))
        p[0, 1] = p1
        p[1, 0] = beta * p1

    print(p)
    return p
