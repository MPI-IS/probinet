from matplotlib import gridspec
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Utils to visualize the data


def plot_hard_membership(graph, communities, pos, node_size, colors, edge_color):
    """
        Plot a graph with nodes colored by their hard memberships.
    """

    plt.figure(figsize=(10, 5))
    for i, k in enumerate(communities):
        plt.subplot(1, 2, i + 1)
        nx.draw_networkx(
            graph,
            pos,
            node_size=node_size,
            node_color=[
                colors[node] for node in communities[k]],
            with_labels=False,
            width=0.5,
            edge_color=edge_color,
            arrows=True,
            arrowsize=5,
            connectionstyle="arc3,rad=0.2")
        plt.title(k, fontsize=17)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def extract_bridge_properties(i, color, U, threshold=0.):
    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [color[c] for c in groups]
    return wedge_sizes, wedge_colors


def plot_soft_membership(graph, thetas, pos, node_size, colors, edge_color):
    """
        Plot a graph with nodes colored by their mixed (soft) memberships.
    """

    plt.figure(figsize=(10, 5))
    for j, k in enumerate(thetas):
        plt.subplot(1, 2, j + 1)
        ax = plt.gca()
        nx.draw_networkx_edges(graph, pos, width=0.5, edge_color=edge_color, arrows=True,
                               arrowsize=5, connectionstyle="arc3,rad=0.2", node_size=150, ax=ax)
        for i, n in enumerate(graph.nodes()):
            wedge_sizes, wedge_colors = extract_bridge_properties(i, colors, thetas[k])
            if len(wedge_sizes) > 0:
                _ = plt.pie(
                    wedge_sizes,
                    center=pos[n],
                    colors=wedge_colors,
                    radius=(
                        node_size[i]) *
                    0.0005)
                ax.axis("equal")
        plt.title(k, fontsize=17)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_adjacency(Bd, M_marginal, M_conditional, nodes, cm='Blues'):  # , sns=None):
    """
        Plot the adjacency matrix and its reconstruction given by the marginal and the conditional expected values.
    """
    # if sns is not None:
    #     sns.set_style('ticks')
    # else:
    sns.set_style('ticks')
    # plt.style.use('seaborn-white')

    plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    plt.subplot(gs[0, 0])
    im = plt.imshow(Bd[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title('Data', fontsize=17)

    plt.subplot(gs[0, 1])
    plt.imshow(M_marginal[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)

    plt.subplot(gs[0, 2])
    plt.imshow(M_conditional[0], vmin=0, vmax=1, cmap=cm)
    plt.xticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.yticks(ticks=np.arange(len(nodes)), labels=nodes(), fontsize=9)
    plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)

    axes = plt.subplot(gs[0, 3])
    cbar = plt.colorbar(im, cax=axes)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


def mapping(G, A):
    old = list(G.nodes)
    new = list(A.nodes)

    mapping = {}
    for x in old:
        mapping[x] = new[x]

    return nx.relabel_nodes(G, mapping)


def plot_graph(
        graph,
        M_marginal,
        M_conditional,
        pos,
        node_size,
        node_color,
        edge_color,
        threshold=0.2):
    """
        Plot a graph and its reconstruction given by the marginal and the conditional expected values.
    """

    plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)

    plt.subplot(gs[0, 0])
    edgewidth = [d['weight'] for (u, v, d) in graph.edges(data=True)]
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
        font_color="black")
    plt.axis('off')
    plt.title('Data', fontsize=17)

    mask = M_marginal[0] < threshold
    M = M_marginal[0].copy()
    M[mask] = 0.
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
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
        edge_cmap=plt.cm.Greys,
        edge_vmin=0,
        edge_vmax=1,
        arrows=True,
        arrowsize=5)
    plt.axis('off')
    plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)

    mask = M_conditional[0] < threshold
    M = M_conditional[0].copy()
    M[mask] = 0.
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    G = mapping(G, graph)
    edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]

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
        edge_cmap=plt.cm.Greys,
        edge_vmin=0,
        edge_vmax=1,
        arrows=True,
        arrowsize=5)
    plt.axis('off')
    plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)

    plt.tight_layout()
    plt.show()


def plot_precision_recall(conf_matrix, cm='Blues'):
    """
        Plot precision and recall of a given confusion matrix.
    """

    plt.figure(figsize=(10, 5))

    # normalized by row
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    plt.subplot(gs[0, 0])
    im = plt.imshow(
        conf_matrix /
        np.sum(
            conf_matrix,
            axis=1)[
            :,
            np.newaxis],
        cmap=cm,
        vmin=0,
        vmax=1)
    plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.yticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.ylabel('True', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    plt.title('Precision', fontsize=17)

    # normalized by column
    plt.subplot(gs[0, 1])
    plt.imshow(conf_matrix / np.sum(conf_matrix, axis=0)[np.newaxis, :], cmap=cm, vmin=0, vmax=1)
    plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
    plt.tick_params(
        axis='y',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False)
    plt.xlabel('Predicted', fontsize=15)
    plt.title('Recall', fontsize=17)

    axes = plt.subplot(gs[0, 2])
    plt.colorbar(im, cax=axes)

    plt.tight_layout()
    plt.show()


def plot_adjacency_samples(Bdata, Bsampled, cm='Blues'):
    """
        Plot the adjacency matrix and five sampled networks.
    """

    plt.figure(figsize=(30, 5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])
    plt.subplot(gs[0, 0])
    plt.imshow(Bdata[0], vmin=0, vmax=1, cmap=cm)
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False)
    plt.title('Data', fontsize=25)

    for i in range(5):
        plt.subplot(gs[0, i + 1])
        plt.imshow(Bsampled[i], vmin=0, vmax=1, cmap=cm)
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False)
        plt.title(f'Sample {i + 1}', fontsize=25)

    plt.tight_layout()
    plt.show()


def plot_A(A, cmap='PuBuGn'):
    """
        Plot the adjacency tensor produced by the generative algorithm.

        INPUT
        ----------
        A : list
            List of scipy sparse matrices, one for each layer.
        cmap : Matplotlib object
               Colormap used for the plot.
    """

    L = len(A)
    for l in range(L):
        Ad = A[l].todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap=plt.get_cmap(cmap))
        ax.set_title(f'Adjacency matrix layer {l}', fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()
