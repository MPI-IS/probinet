"""
Base classes for synthetic network generation.
"""

import logging
import math
from abc import ABCMeta
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from probinet.visualization.plot import plot_A

from ..evaluation.expectation_computation import compute_mean_lambda0
from ..input.stats import print_graph_stats
from ..utils.matrix_operations import normalize_nonzero_membership
from ..utils.tools import log_and_raise_error, output_adjacency

DEFAULT_N = 1000
DEFAULT_L = 1
DEFAULT_K = 2
DEFAULT_ETA = 50
DEFAULT_ALPHA_HL = 6
DEFAULT_AVG_DEGREE = 15
DEFAULT_STRUCTURE = "assortative"

DEFAULT_PERC_OVERLAPPING = 0.2
DEFAULT_CORRELATION_U_V = 0.0
DEFAULT_ALPHA = 0.1

DEFAULT_SEED = 10
DEFAULT_IS_SPARSE = True

DEFAULT_SHOW_DETAILS = True
DEFAULT_SHOW_PLOTS = False
DEFAULT_OUTPUT_NET = False


class Structure(Enum):
    ASSORTATIVE = "assortative"
    DISASSORTATIVE = "disassortative"
    CORE_PERIPHERY = "core-periphery"
    DIRECTED_BIASED = "directed-biased"


class GraphProcessingMixin:
    """
    Mixin class for graph processing and evaluation methods.
    """

    def _remove_nodes(self, graph, nodes_to_remove):
        """
        Remove nodes from the graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            The graph from which nodes will be removed.
        nodes_to_remove : list
            List of nodes to remove from the graph.
        """
        graph.remove_nodes_from(nodes_to_remove)

    def _update_nodes_list(self, graph):
        """
        Update the list of nodes in the graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            The graph from which to get the list of nodes.

        Returns
        -------
        list
            List of nodes in the graph.
        """
        return list(graph.nodes())

    def _append_layer_graph(self, graph, nodes, layer_graphs):
        """
        Append the layer graph to the list of layer graphs.

        Parameters
        ----------
        graph : networkx.DiGraph
            The graph to convert to a sparse array.
        nodes : list
            List of nodes in the graph.
        layer_graphs : list
            List to which the layer graph will be appended.
        """
        layer_graphs.append(nx.to_scipy_sparse_array(graph, nodelist=nodes))

    def _validate_parameters_shape(self, u, v, w, N, K, L):
        """
        Validate the shape of the parameters u, v, and w.

        Parameters
        ----------
        u : ndarray
            Outgoing membership matrix.
        v : ndarray
            Incoming membership matrix.
        w : ndarray
            Affinity tensor.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        L : int
            Number of layers.

        Raises
        ------
        ValueError
            If the shape of any parameter is incorrect.
        """
        if u.shape != (N, K):
            log_and_raise_error(
                ValueError, "The shape of the parameter u has to be (N, K)."
            )

        if v.shape != (N, K):
            log_and_raise_error(
                ValueError, "The shape of the parameter v has to be (N, K)."
            )

        if w.shape != (L, K, K):
            log_and_raise_error(
                ValueError, "The shape of the parameter w has to be (L, K, K)."
            )

    def _output_parameters(self) -> None:
        """
        Output results in a compressed file.
        """

        self.out_folder.mkdir(parents=True, exist_ok=True)

        output_parameters = self.out_folder / (
            "theta" + (self.end_file if self.end_file else "")
        )  # Save parameters
        output_params = {
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "nodes": self.nodes,
        }

        # Add eta if it exists
        if hasattr(self, "eta"):
            output_params["eta"] = self.eta

        # Add beta if it exists
        if hasattr(self, "beta"):
            output_params["beta"] = self.beta

        # Save parameters
        np.savez_compressed(output_parameters.with_suffix(".npz"), **output_params)

        logging.info("Parameters saved in: %s", output_parameters.with_suffix(".npz"))
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _plot_matrix(self, M: np.ndarray, L: int, cmap: str = "PuBuGn") -> None:
        """
        Plot the marginal means matrix.

        Parameters
        ----------
        M : ndarray
            Mean lambda for all entries.
        L : int
            Number of layers.
        cmap : str, optional
            Colormap used for the plot, by default "PuBuGn".
        """
        for layer in range(L):
            np.fill_diagonal(M[layer], 0.0)
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(M[layer], cmap=plt.get_cmap(cmap))
            ax.set_title(f"Marginal means matrix layer {layer}", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    def _extract_edge_data(self, edges: list) -> list:
        """
        Extract edge data from a list of edges.

        Parameters
        ----------
        edges : list
            List of edges with data.

        Returns
        -------
        list
            List of edges with source, target, and weight.
        """
        try:
            data = [[u, v, d["weight"]] for u, v, d in edges]
        except KeyError:
            data = [[u, v, 1] for u, v, d in edges]
        return data

    def output_adjacency(
        self, G: nx.MultiDiGraph, outfile: Optional[str] = None
    ) -> None:
        """
        Output the adjacency matrix. Default format is space-separated .csv with 3 columns:
        node1 node2 weight

        Parameters
        ----------
        G: MultiDiGraph
           MultiDiGraph NetworkX object.
        outfile: str, optional
                 Name of the adjacency matrix.
        """

        # Create a Path object for the evaluation directory
        out_folder_path = Path(self.out_folder)

        # Create evaluation dir if it does not exist
        out_folder_path.mkdir(parents=True, exist_ok=True)

        # Check if the evaluation file name is provided
        if outfile is None:
            # If not provided, generate a default file name using the seed and average degree
            outfile = "syn" + str(self.seed) + "_k" + str(int(self.avg_degree)) + ".dat"

        # Get the list of edges from the graph along with their data
        edges = list(G.edges(data=True))
        data = self._extract_edge_data(edges)

        # Create a DataFrame from the edge data
        df = pd.DataFrame(data, columns=["source", "target", "w"], index=None)

        # Save the DataFrame to a CSV file
        df.to_csv(self.out_folder + outfile, index=False, sep=" ")

        logging.info("Adjacency matrix saved in: %s", self.out_folder + outfile)


class BaseSyntheticNetwork(metaclass=ABCMeta):
    """
    A base abstract class for generation and management of synthetic networks.

    Suitable for representing any type of synthetic network.
    """

    def __init__(
        self,
        N: int = DEFAULT_N,
        L: int = DEFAULT_L,
        K: int = DEFAULT_K,
        seed: int = DEFAULT_SEED,
        eta: float = DEFAULT_ETA,
        out_folder: Optional[PathLike] = None,
        output_parameters: bool = DEFAULT_OUTPUT_NET,
        output_adj: bool = DEFAULT_OUTPUT_NET,
        outfile_adj: Optional[str] = None,
        end_file: Optional[str] = None,
        show_details: bool = DEFAULT_SHOW_DETAILS,
        show_plots: bool = DEFAULT_SHOW_PLOTS,
        **kwargs,  # these kwargs are needed later on
    ):
        """
        Initialize the base synthetic network with the given parameters.

        Parameters
        ----------
        N : int, optional
            Number of nodes in the network (default is DEFAULT_N).
        L : int, optional
            Number of layers in the network (default is DEFAULT_L).
        K : int, optional
            Number of communities in the network (default is DEFAULT_K).
        seed : int, optional
            Seed for the random number generator (default is DEFAULT_SEED).
        eta : float, optional
            Reciprocity coefficient (default is DEFAULT_ETA).
        out_folder : str, optional
            Path to the evaluation folder (default is DEFAULT_OUT_FOLDER).
        output_parameters : bool, optional
            Flag to save the network (default is DEFAULT_OUTPUT_NET).
        output_adj : bool, optional
            Flag to save the adjacency matrix (default is DEFAULT_OUTPUT_NET).
        show_details : bool, optional
            Flag to print graph statistics (default is DEFAULT_SHOW_DETAILS).
        show_plots : bool, optional
            Flag to plot the network (default is DEFAULT_SHOW_PLOTS).
        kwargs : dict
            Additional keyword arguments for further customization.
        """

        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities

        # Set seed random number generator
        self.seed = seed
        self.eta = eta
        self.rng = np.random.default_rng(seed)

        self.out_folder = out_folder
        self.output_parameters = output_parameters
        self.output_adj = output_adj
        self.outfile_adj = outfile_adj
        self.end_file = end_file

        self.show_details = show_details
        self.show_plots = show_plots


class StandardMMSBM(BaseSyntheticNetwork, GraphProcessingMixin):
    """
    Create a synthetic, directed, and weighted network (possibly multilayer)
    by a standard mixed-membership stochastic block-models.

    - It models marginals (iid assumption) with Poisson distributions
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__doc__ = BaseSyntheticNetwork.__init__.__doc__

        parameters = kwargs.get("parameters")

        self.init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        if self.output_parameters:
            self._output_parameters()
        if self.output_adj:
            output_adjacency(self.layer_graphs, self.out_folder, self.outfile_adj)

        if self.show_details:
            print_graph_stats(self.G)
        if self.show_plots:
            plot_A(self.layer_graphs)
            if self.M is not None:
                self._plot_M()

    def init_mmsbm_params(self, **kwargs) -> None:
        """
        Check MMSBM-specific parameters
        """

        super().__init__(**kwargs)

        if "avg_degree" in kwargs:
            avg_degree = kwargs["avg_degree"]
            if avg_degree <= 0:  # (in = out) average degree
                log_and_raise_error(
                    ValueError, "The average degree has to be greater than 0.!"
                )
        else:
            message = f"avg_degree parameter was not set. Defaulting to avg_degree={DEFAULT_AVG_DEGREE}"
            logging.warning(message)
            avg_degree = DEFAULT_AVG_DEGREE
        self.avg_degree = avg_degree
        self.ExpEdges = int(self.avg_degree * self.N * 0.5)

        if "is_sparse" in kwargs:
            is_sparse = kwargs["is_sparse"]
        else:
            message = f"is_sparse parameter was not set. Defaulting to is_sparse={DEFAULT_IS_SPARSE}"
            logging.warning(message)
            is_sparse = DEFAULT_IS_SPARSE
        self.is_sparse = is_sparse

        if "outfile_adj" in kwargs:
            outfile_adj = kwargs["outfile_adj"]
        else:
            try:
                message = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_eta_seed"
                logging.warning(message)
                outfile_adj = "_".join(
                    [
                        str(),
                        str(self.N),
                        str(self.L),
                        str(self.K),
                        str(self.avg_degree),
                        str(self.eta),
                        str(self.seed),
                    ]
                )
            except AttributeError:
                message = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_seed"
                logging.warning(message)
                outfile_adj = "_".join(
                    [
                        str(),
                        str(self.N),
                        str(self.L),
                        str(self.K),
                        str(self.avg_degree),
                        str(self.seed),
                    ]
                )
        self.outfile_adj = outfile_adj  # Formerly self.label

        # SETUP overlapping communities

        if "perc_overlapping" in kwargs:
            perc_overlapping = kwargs["perc_overlapping"]
            if (perc_overlapping < 0) or (
                perc_overlapping > 1
            ):  # fraction of nodes with mixed membership
                log_and_raise_error(
                    ValueError,
                    "The percentage of overlapping nodes has to be in  [0, 1]!",
                )
        else:
            message = (
                f"perc_overlapping parameter was not set. Defaulting to perc_overlapping"
                f"={DEFAULT_PERC_OVERLAPPING}"
            )
            logging.warning(message)
            perc_overlapping = DEFAULT_PERC_OVERLAPPING
        self.perc_overlapping = perc_overlapping

        if self.perc_overlapping:
            # correlation between u and v synthetically generated
            if "correlation_u_v" in kwargs:
                correlation_u_v = kwargs["correlation_u_v"]
                if (correlation_u_v < 0) or (correlation_u_v > 1):
                    log_and_raise_error(
                        ValueError,
                        "The correlation between u and v has to be in [0, 1]!",
                    )
            else:
                message = (
                    f"correlation_u_v parameter for overlapping communities was not set. "
                    f"Defaulting to corr={DEFAULT_CORRELATION_U_V}"
                )
                logging.warning(message)
                correlation_u_v = DEFAULT_CORRELATION_U_V
            self.correlation_u_v = correlation_u_v

            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            else:
                message = (
                    f"alpha parameter of Dirichlet distribution was not set. "
                    f"Defaulting to alpha={[DEFAULT_ALPHA] * self.K}"
                )
                logging.warning(message)
                alpha = [DEFAULT_ALPHA] * self.K
            if isinstance(alpha, float):
                if alpha <= 0:
                    log_and_raise_error(
                        ValueError,
                        "Each entry of the Dirichlet parameter has to be positive!",
                    )

                alpha = [alpha] * self.K
            elif len(alpha) != self.K:
                log_and_raise_error(
                    ValueError, "The parameter alpha should be a list of length K."
                )
            if not all(alpha):
                log_and_raise_error(
                    ValueError,
                    "Each entry of the Dirichlet parameter has to be positive!",
                )
            self.alpha = alpha

        # SETUP informed structure

        if "structure" in kwargs:
            structure = kwargs["structure"]
        else:
            message = (
                f"structure parameter was not set. Defaulting to "
                f"structure={[DEFAULT_STRUCTURE] * self.L}"
            )
            logging.warning(message)
            structure = [DEFAULT_STRUCTURE] * self.L
        if isinstance(structure, str):
            if structure not in ["assortative", "disassortative"]:
                message = (
                    "The available structures for the affinity tensor w are: "
                    "assortative, disassortative!"
                )
                log_and_raise_error(ValueError, message)
            structure = [structure] * self.L
        elif len(structure) != self.L:
            message = (
                "The parameter structure should be a list of length L. "
                "Each entry defines the structure of the corresponding layer!"
            )
            log_and_raise_error(ValueError, message)
        for e in structure:
            if e not in ["assortative", "disassortative"]:
                message = (
                    "The available structures for the affinity tensor w are: "
                    "assortative, disassortative!"
                )
                log_and_raise_error(ValueError, message)
        self.structure = structure

    def build_Y(
        self, parameters: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Generate network layers G using the latent variables,
        with the generative models A_ij ~ P(A_ij|u,v,w)

        Parameters
        ----------
        parameters: Tuple[np.ndarray, np.ndarray, np.ndarray], optional
        """

        # Latent variables

        if parameters is None:
            # generate latent variables
            self.u, self.v, self.w = self._generate_lv()
        else:
            # set latent variables
            self.u, self.v, self.w = parameters
            self._validate_parameters_shape(
                self.u, self.v, self.w, self.N, self.K, self.L
            )

        # Generate Y

        self.M = compute_mean_lambda0(self.u, self.v, self.w)
        for layer in range(self.L):
            np.fill_diagonal(self.M[layer], 0)
        # sparsity parameter for Y
        if self.is_sparse:
            c = self.ExpEdges / self.M.sum()
            self.M *= c
            if parameters is None:
                self.w *= c

        Y = self.rng.poisson(self.M)

        # Create networkx DiGraph objects for each layer for easier manipulation

        nodes_to_remove = []
        self.G = []
        for layer in range(self.L):
            self.G.append(nx.from_numpy_array(Y[layer], create_using=nx.DiGraph()))
            Gc = max(nx.weakly_connected_components(self.G[layer]), key=len)
            nodes_to_remove.append(set(self.G[layer].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        self.layer_graphs: list[np.ndarray] = []
        for layer in range(self.L):
            self._remove_nodes(self.G[layer], list(n_to_remove))
            self.nodes = self._update_nodes_list(self.G[layer])
            self._append_layer_graph(self.G[layer], self.nodes, self.layer_graphs)

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)

    def _apply_overlapping(
        self, u: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce overlapping membership in the NxK membership vectors u and v, by using a
        Dirichlet distribution.

        INPUT, OUTPUT
        ----------
        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
        """

        # number of nodes belonging to more communities
        overlapping = int(self.N * self.perc_overlapping)
        ind_over = self.rng.integers(low=0, high=len(u), size=overlapping)

        u[ind_over] = self.rng.dirichlet(self.alpha * np.ones(self.K), overlapping)
        v[ind_over] = self.correlation_u_v * u[ind_over] + (
            1.0 - self.correlation_u_v
        ) * self.rng.dirichlet(self.alpha * np.ones(self.K), overlapping)
        if self.correlation_u_v == 1.0:
            assert np.allclose(u, v)
        if self.correlation_u_v > 0:
            v = normalize_nonzero_membership(v)

        return u, v

    def _sample_membership_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the NxK membership vectors u and v without overlapping.

        Returns
        ----------
        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
        """

        # Generate equal-size unmixed group membership
        size = int(self.N / self.K)
        u = np.zeros((self.N, self.K))
        v = np.zeros((self.N, self.K))
        for i in range(self.N):
            q = int(math.floor(float(i) / float(size)))
            if q == self.K:
                u[i:, self.K - 1] = 1.0
                v[i:, self.K - 1] = 1.0
            else:
                for j in range(q * size, q * size + size):
                    u[j, q] = 1.0
                    v[j, q] = 1.0

        return u, v

    def _compute_affinity_matrix(self, structure: str, a: float = 0.1) -> np.ndarray:
        """
        Compute the KxK affinity matrix w with probabilities between and within groups.

        Parameters
        ----------
        structure : list
                    List of structure of network layers.
        a : float
            Parameter for secondary probabilities.

        Parameters
        -------
        p : Numpy array
            Array with probabilities between and within groups. Element (k,h)
            gives the density of edges going from the nodes of group k to nodes of group h.
        """

        p1 = self.avg_degree * self.K / self.N

        if structure == "assortative":
            p = p1 * a * np.ones((self.K, self.K))  # secondary-probabilities
            np.fill_diagonal(p, p1 * np.ones(self.K))  # primary-probabilities

        elif structure == "disassortative":
            p = p1 * np.ones((self.K, self.K))  # primary-probabilities
            np.fill_diagonal(p, a * p1 * np.ones(self.K))  # secondary-probabilities

        return p

    def _generate_lv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate latent variables for a MMSBM, assuming network layers are independent
        and communities are shared across layers.

        Returns
        -------
        u : ndarray
            Outgoing membership matrix.
        v : ndarray
            Incoming membership matrix.
        w : ndarray
            Affinity tensor.

        """

        # Generate u, v
        u, v = self._sample_membership_vectors()
        # Introduce the overlapping membership
        if self.perc_overlapping > 0:
            u, v = self._apply_overlapping(u, v)

        # Generate w
        w = np.zeros((self.L, self.K, self.K))
        for layer in range(self.L):
            w[layer, :, :] = self._compute_affinity_matrix(self.structure[layer])

        return u, v, w

    def _plot_M(self, cmap: str = "PuBuGn") -> None:
        """
        Plot the marginal means produced by the generative algorithm.

        Parameters
        ----------
        M : ndarray
            Mean lambda for all entries.
        """
        self._plot_matrix(self.M, self.L, cmap)


def affinity_matrix(
    structure: Union[Structure, str] = Structure.ASSORTATIVE.value,
    N: int = 100,
    K: int = 2,
    avg_degree: float = 4.0,
    a: float = 0.1,
    b: float = 0.3,
) -> np.ndarray:
    """
    Compute the KxK affinity matrix w with probabilities between and within groups.

    Parameters
    ----------
    structure : Structure, optional
        Structure of the network (default is Structure.ASSORTATIVE).
    N : int, optional
        Number of nodes (default is 100).
    K : int, optional
        Number of communities (default is 2).
    avg_degree : float, optional
        Average degree of the network (default is 4.0).
    a : float, optional
        Parameter for secondary probabilities (default is 0.1).
    b : float, optional
        Parameter for secondary probabilities in 'core-periphery' and 'directed-biased' structures (default is 0.3).

    Returns
    -------
    np.ndarray
        KxK affinity matrix. Element (k,h) gives the density of edges going from the nodes of group k to nodes of group h.
    """

    # Adjust b based on a
    b *= a

    # Calculate primary probability
    p1 = avg_degree * K / N

    # Initialize the affinity matrix based on the structure
    if structure == Structure.ASSORTATIVE.value:
        # Assortative structure: higher probability within groups
        p = p1 * a * np.ones((K, K))  # secondary probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary probabilities

    elif structure == Structure.DISASSORTATIVE.value:
        # Disassortative structure: higher probability between groups
        p = p1 * np.ones((K, K))  # primary probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary probabilities

    elif structure == Structure.CORE_PERIPHERY.value:
        # Core-periphery structure: core nodes have higher probability
        p = p1 * np.ones((K, K))
        np.fill_diagonal(np.fliplr(p), a * p1)
        p[1, 1] = b * p1

    elif structure == Structure.DIRECTED_BIASED.value:
        # Directed-biased structure: directional bias in probabilities
        p = a * p1 * np.ones((K, K))
        p[0, 1] = p1
        p[1, 0] = b * p1

    else:
        raise ValueError("Invalid structure type.")

    return p
