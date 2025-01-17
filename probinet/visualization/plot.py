"""
It provides a set of plotting functions for visualizing the results of the generative models.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import colormaps, gridspec
from matplotlib.ticker import MaxNLocator


def plot_hard_membership(
    graph: nx.DiGraph,
    communities: Dict,
    pos: Dict,
    node_size: np.ndarray,
    colors: Dict,
    edge_color: str,
) -> plt.Figure:
    """
    Plot a graph with nodes colored by their hard memberships.

    Parameters
    ----------
    graph : nx.DiGraph
            Graph to be plotted.
    communities : Dict
                  Dictionary with the communities.
    pos : Dict
            Dictionary with the positions of the nodes.
    node_size : ndarray
                Array with the sizes of the nodes.
    colors : Dict
             Dictionary with the colors of the nodes.
    edge_color : str
                 Color of the edges.

    Returns
    -------
    fig : plt.Figure
          The matplotlib figure object.
    """

    fig = plt.figure(figsize=(8, 3))
    for i, k in enumerate(communities):
        plt.subplot(1, 2, i + 1)
        nx.draw_networkx(
            graph,
            pos,
            node_size=node_size,
            node_color=[colors[node] for node in communities[k]],
            with_labels=False,
            width=0.5,
            edge_color=edge_color,
            arrows=True,
            arrowsize=5,
            connectionstyle="arc3,rad=0.2",
        )
        plt.title(rf"${k}$", fontsize=17)
        plt.axis("off")
    plt.tight_layout()

    return fig


def extract_bridge_properties(
    i: int, color: dict, U: np.ndarray, threshold: float = 0.2
) -> Tuple[np.ndarray, list]:
    """
    Extract the properties of the bridges of a node i.

    Parameters
    ----------
    i : int
        Index of the node.
    color : dict
            Dictionary with the colors of the nodes.
    U : ndarray
        Out-going membership matrix.
    threshold : float
                Threshold for the membership values.
    Returns
    -------
    wedge_sizes : ndarray
                  Sizes of the wedges.
    wedge_colors : list
                   Colors of the wedges.
    """

    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [color[c] for c in groups]
    return wedge_sizes, wedge_colors


def plot_soft_membership(
    graph: nx.DiGraph,
    thetas: Dict,
    pos: Dict,
    node_size: np.ndarray,
    colors: Dict,
    edge_color: str,
) -> plt.Figure:
    """
    Plot a graph with nodes colored by their mixed (soft) memberships.

    Parameters
    ----------
    graph : nx.DiGraph
            Graph to be plotted.
    thetas : Dict
             Dictionary with the mixed memberships.
    pos : Dict
            Dictionary with the positions of the nodes.
    node_size : ndarray
                Array with the sizes of the nodes.
    colors : Dict
             Dictionary with the colors of the nodes.
    edge_color : str
                 Color of the edges.

    Returns
    -------
    fig : plt.Figure
          The matplotlib figure object.
    """

    fig = plt.figure(figsize=(9, 4))
    for j, k in enumerate(thetas):
        plt.subplot(1, 2, j + 1)
        ax = plt.gca()
        nx.draw_networkx_edges(
            graph,
            pos,
            width=0.5,
            edge_color=edge_color,
            arrows=True,
            arrowsize=5,
            connectionstyle="arc3,rad=0.2",
            node_size=150,
            ax=ax,
        )
        for i, n in enumerate(graph.nodes()):
            wedge_sizes, wedge_colors = extract_bridge_properties(i, colors, thetas[k])
            if len(wedge_sizes) > 0:
                _ = plt.pie(
                    wedge_sizes,
                    center=pos[n],
                    colors=wedge_colors,
                    radius=(node_size[i]) * 0.0005,
                )
                ax.axis("equal")
        plt.title(rf"${k}$", fontsize=17)
        plt.axis("off")
    plt.tight_layout()

    return fig


def plot_adjacency(
    Bd: np.ndarray,
    M_marginal: np.ndarray,
    M_conditional: np.ndarray,
    nodes: List,
    cm: str = "Blues",
) -> plt.Figure:
    """
    Plot the adjacency matrix and its reconstruction given by the marginal and the conditional
    expected values.

    Parameters
    ----------
    Bd : ndarray
         Adjacency matrix.
    M_marginal : ndarray
                 Marginal expected values.
    M_conditional : ndarray
                    Conditional expected values.
    nodes : list
            List of nodes.
    cm : Matplotlib object
         Colormap used for the plot.

    Returns
    -------
    fig : plt.Figure
          The matplotlib figure object.
    """

    sns.set_style("ticks")

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    plt.subplot(gs[0, 0])
    im = plt.imshow(Bd[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.title("Data", fontsize=17)

    plt.subplot(gs[0, 1])
    plt.imshow(M_marginal[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.title(r"$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$", fontsize=17)

    plt.subplot(gs[0, 2])
    plt.imshow(M_conditional[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes, fontsize=9)
    plt.title(r"$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$", fontsize=17)

    axes = plt.subplot(gs[0, 3])
    cbar = plt.colorbar(im, cax=axes)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()

    return fig


def mapping(G: nx.DiGraph, A: nx.DiGraph) -> nx.DiGraph:
    """
    Map the nodes of a graph G to the nodes of a graph A.

    Parameters
    ----------
    G : nx.DiGraph
        Graph to be mapped.
    A : nx.DiGraph
        Graph to be mapped to.
    Returns
    -------
    G : nx.DiGraph
        Graph G with the nodes mapped to the nodes of A.
    """

    # Define the mapping
    old = list(G.nodes)
    new = list(A.nodes)

    mapping_dict = {}
    for x in old:
        mapping_dict[x] = new[x]
    # Return the mapped graph
    return nx.relabel_nodes(G, mapping_dict)


def plot_graph(
    graph: nx.DiGraph,
    M_marginal: np.ndarray,
    M_conditional: np.ndarray,
    pos: Dict,
    node_size: int,
    node_color: str,
    edge_color: str,
    threshold: float = 0.2,
) -> plt.Figure:
    """
    Plot a graph and its reconstruction given by the marginal and the conditional expected values.

    Parameters
    ----------
    graph : nx.DiGraph
            Graph to be plotted.
    M_marginal : ndarray
                 Marginal expected values.
    M_conditional : ndarray
                    Conditional expected values.
    pos : dict
          Dictionary with the positions of the nodes.
    node_size : int
                Size of the nodes.
    node_color : str
                 Color of the nodes.
    edge_color : str
                 Color of the edges.
    threshold : float
                Threshold for the membership values.

    Returns
    -------
    fig : plt.Figure
          The matplotlib figure object.
    """

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)

    plt.subplot(gs[0, 0])
    edgewidth = [d["weight"] for (u, v, d) in graph.edges(data=True)]
    nx.draw_networkx(
        graph,
        pos,
        node_size=node_size,
        node_color=node_color,
        connectionstyle="arc3,rad=0.2",
        with_labels=False,
        width=edgewidth,
        edge_color=edge_color,
        arrows=True,
        arrowsize=5,
        font_size=15,
        font_color="black",
    )
    plt.axis("off")
    plt.title("Data", fontsize=17)

    mask = M_marginal[0] < threshold
    M = M_marginal[0].copy()
    M[mask] = 0.0
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d["weight"] for (u, v, d) in G.edges(data=True)]
    plt.subplot(gs[0, 1])
    nx.draw_networkx(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        connectionstyle="arc3,rad=0.2",
        with_labels=False,
        width=edgewidth,
        edge_color=edgewidth,
        edge_cmap=colormaps["Greys"],
        edge_vmin=0,
        edge_vmax=1,
        arrows=True,
        arrowsize=5,
    )
    plt.axis("off")
    plt.title(r"$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$", fontsize=17)

    mask = M_conditional[0] < threshold
    M = M_conditional[0].copy()
    M[mask] = 0.0
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d["weight"] for (u, v, d) in G.edges(data=True)]

    plt.subplot(gs[0, 2])
    nx.draw_networkx(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        connectionstyle="arc3,rad=0.2",
        with_labels=False,
        width=edgewidth,
        edge_color=edgewidth,
        edge_cmap=colormaps["Greys"],
        edge_vmin=0,
        edge_vmax=1,
        arrows=True,
        arrowsize=5,
    )
    plt.axis("off")
    plt.title(r"$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$", fontsize=17)

    plt.tight_layout()

    return fig


def plot_precision_recall(conf_matrix: np.ndarray, cm: str = "Blues") -> plt.Figure:
    """
    Plot precision and recall of a given confusion matrix.

    Parameters
    ----------
    conf_matrix : ndarray
                  Confusion matrix.
    cm : Matplotlib object
         Colormap used for the plot.

    Returns
    -------
    fig : plt.Figure
          The matplotlib figure object.
    """

    fig = plt.figure(figsize=(8, 3))

    # normalized by row
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    plt.subplot(gs[0, 0])
    im = plt.imshow(
        conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis],
        cmap=cm,
        vmin=0,
        vmax=1,
    )
    plt.xticks(
        [0, 1, 2, 3], labels=["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"], fontsize=13
    )
    plt.yticks(
        [0, 1, 2, 3], labels=["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"], fontsize=13
    )
    plt.ylabel("True", fontsize=15)
    plt.xlabel("Predicted", fontsize=15)
    plt.title("Precision", fontsize=17)

    # normalized by column
    plt.subplot(gs[0, 1])
    plt.imshow(
        conf_matrix / np.sum(conf_matrix, axis=0)[np.newaxis, :],
        cmap=cm,
        vmin=0,
        vmax=1,
    )
    plt.xticks(
        [0, 1, 2, 3], labels=["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"], fontsize=13
    )
    plt.tick_params(
        axis="y",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    plt.xlabel("Predicted", fontsize=15)
    plt.title("Recall", fontsize=17)

    axes = plt.subplot(gs[0, 2])
    plt.colorbar(im, cax=axes)

    # plt.tight_layout()

    return fig


def plot_adjacency_samples(
    Bdata: List, Bsampled: List, cm: str = "Blues"
) -> plt.Figure:
    """
    Plot the adjacency matrix and five sampled networks.

    Parameters
    ----------
    Bdata : list
        List of adjacency matrices for the data.
    Bsampled : list
        List of adjacency matrices for sampled networks.
    cm : Matplotlib object
        Colormap used for the plot.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object.
    """

    fig = plt.figure(figsize=(30, 5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])
    plt.subplot(gs[0, 0])
    plt.imshow(Bdata[0], vmin=0, vmax=1, cmap=cm)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    plt.title("Data", fontsize=25)

    for i in range(5):
        plt.subplot(gs[0, i + 1])
        plt.imshow(Bsampled[i], vmin=0, vmax=1, cmap=cm)
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False,
        )
        plt.title(f"Sample {i + 1}", fontsize=25)

    plt.tight_layout()

    return fig


def plot_A(A: List, cmap: str = "Blues") -> List[plt.Figure]:
    """
    Plot the adjacency tensor produced by the generative algorithm.

    Parameters
    ----------
    A : list
        List of scipy sparse matrices, one for each layer.
    cmap : Matplotlib object
           Colormap used for the plot.
    Returns
    -------
    figures : list
        List of matplotlib figure objects.
    """

    figures = []
    L = len(A)
    for layer in range(L):
        Ad = A[layer].todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap=plt.get_cmap(cmap))
        ax.set_title(f"Adjacency matrix layer {layer}", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)  # pylint: disable=undefined-loop-variable
        figures.append(fig)

    return figures


def plot_L(
    values: List,
    indices: Optional[List] = None,
    k_i: int = 0,  # 5 for ACD
    xlab: str = "Iterations",
    ylabel: str = "Log-likelihood values",
    figsize: tuple = (10, 5),  # (7, 7) for ACD
    int_ticks: bool = False,
) -> plt.Figure:
    """
    Plot the log-likelihood.
    Parameters
    ----------
    values : list
             List of log-likelihood values.
    indices : list
              List of indices.
    k_i : int
            Number of initial iterations to be ignored.
    xlab : str
             Label of the x-axis.
    ylabel : str

    figsize : tuple
              Figure size.
    int_ticks : bool
                Flag to use integer ticks.
    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object.
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylabel)
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    return fig


def plot_M(
    M: np.ndarray,
    cmap: str = "PuBuGn",
    figsize: Tuple[int, int] = (7, 7),
    fontsize: int = 15,
) -> None:
    """
    Plot the M matrix produced by the generative algorithm. Each entry is the
    Poisson mean associated with each pair of nodes in the graph.

    Parameters
    ----------
    M : np.ndarray
        NxN M matrix associated with the graph. Contains all the means used
        for generating edges.
    cmap : str, optional
        Colormap used for the plot.
    figsize : Tuple[int, int], optional
        Size of the figure to be plotted.
    fontsize : int, optional
        Font size to be used in the plot title.
    """

    _, ax = plt.subplots(figsize=figsize)
    ax.matshow(M, cmap=plt.get_cmap(cmap))
    ax.set_title("MT means matrix", fontsize=fontsize)
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)
