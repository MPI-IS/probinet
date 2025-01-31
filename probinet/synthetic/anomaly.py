"""
Class for generation and management of synthetic networks with anomalies
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import brentq

from ..models.constants import OUTPUT_FOLDER
from ..synthetic.base import GraphProcessingMixin, affinity_matrix
from ..synthetic.dynamic import eq_c, membership_vectors
from ..utils.tools import flt, get_or_create_rng
from ..visualization.plot import plot_M

EPS = 1e-12  # Small value to avoid division by zero


class SyntNetAnomaly(GraphProcessingMixin):
    """
    Class for generation and management of synthetic networks with anomalies.
    """

    def __init__(
        self,
        m: int = 1,
        N: int = 100,
        K: int = 2,
        avg_degree: float = 4.0,
        rho_anomaly: float = 0.1,
        structure: str = "assortative",
        label: Optional[str] = None,
        pi: float = 0.8,
        eta: float = 0.5,
        L1: bool = False,
        ag: float = 0.6,
        bg: float = 1.0,
        corr: float = 0.0,
        over: float = 0.0,
        verbose: int = 0,
        out_folder: Path = OUTPUT_FOLDER,
        output_parameters: bool = False,
        output_adj: bool = False,
        outfile_adj: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize the SyntNetAnomaly class.

        Parameters
        ----------
        m
            Number of networks to be generated (default is 1).
        N
            Network size (number of nodes) (default is 100).
        K
            Number of communities (default is 2).
        avg_degree
            Required average degree of the network (default is 4.0).
        rho_anomaly
            Proportion of anomalies in the network (default is 0.1).
        structure
            Structure of the affinity matrix (default is "assortative").
        label
            Label associated with the set of inputs (default is None).
        pi
            Binomial parameter for edge generation (default is 0.8).
        eta
            Parameter of the Dirichlet distribution (default is 0.5).
        L1
            Flag for L1 norm (default is False).
        ag
            Alpha parameter of the Gamma distribution (default is 0.6).
        bg
            Beta parameter of the Gamma distribution (default is 1.0).
        corr
            Correlation between u and v synthetically generated (default is 0.0).
        over
            Fraction of nodes with mixed membership (default is 0.0).
        verbose
            Verbosity level (default is 0).
        folder
            Folder path for saving outputs (default is "").
        output_parameters
            Flag for storing the parameters (default is False).
        output_adj
            Flag for storing the generated adjacency matrix (default is False).
        outfile_adj
            Name for saving the adjacency matrix (default is None).
        rng
            Random number generator (default is None).

        Raises
        ------
        ValueError
            If any of the input parameters are out of their valid ranges.
        """
        # Set network size (node number)
        self.N = N
        # Set number of communities
        self.K = K
        # Set number of networks to be generated
        self.m = m
        # Set seed random number generator
        self.rng = get_or_create_rng(rng)
        # Set label (associated uniquely with the set of inputs)
        if label is not None:
            self.label = label
        else:
            self.label = ("_").join(
                [str(N), str(K), str(avg_degree), str(flt(rho_anomaly, d=2))]
            )
        # Initialize data folder path
        self.out_folder = out_folder
        # Set flag for storing the parameters
        self.output_parameters = output_parameters
        # Set flag for storing the generated adjacency matrix
        self.output_adj = output_adj
        # Set name for saving the adjacency matrix
        self.outfile_adj = outfile_adj
        # Set required average degree
        self.avg_degree = avg_degree
        self.rho_anomaly = rho_anomaly

        # Set verbosity flag
        if verbose > 2 and not isinstance(verbose, int):
            raise ValueError(
                "The verbosity parameter can only assume values in {0,1,2}!"
            )
        self.verbose = verbose

        # Set Bernoullis parameters
        # if mu < 0 or mu > 1:
        # raise ValueError('The Binomial parameter mu has to be in [0, 1]!')

        # Check if the value of pi is within the valid range [0, 1]
        if pi < 0 or pi > 1:
            # If not, raise a ValueError with a descriptive message
            raise ValueError("The Binomial parameter pi has to be in [0, 1]!")

        # If pi is exactly 1, subtract a very small value to avoid issues with calculations
        if np.isclose(pi, 1, atol=EPS):
            pi = 1 - EPS

        # If pi is exactly 0, add a very small value to avoid issues with calculations
        if np.isclose(pi, 0, atol=EPS):
            pi = EPS

        # Assign the adjusted value of pi to the instance variable self.pi
        self.pi = pi

        # Check if the value of rho_anomaly is within the valid range [0, 1]
        if rho_anomaly < 0 or rho_anomaly > 1:
            # If not, raise a ValueError with a descriptive message
            raise ValueError("The rho anomaly has to be in [0, 1]!")

        # Calculate the expected number of edges in the network
        self.ExpM = self.avg_degree * self.N * 0.5

        # Calculate the proportion of reciprocal edges in the network
        mu = (
            self.rho_anomaly
            * self.ExpM
            / ((1 - np.exp(-self.pi)) * (self.N**2 - self.N))
        )

        # If mu is exactly 1, subtract a very small value to avoid issues with calculations
        if mu == 1:
            mu = 1 - EPS

        # If mu is exactly 0, add a very small value to avoid issues with calculations
        if mu == 0:
            mu = EPS

        # Assert that mu is within the valid range (0, 1)
        assert 1.0 > mu > 0.0, "mu has to be in (0, 1)!"

        # Assign the adjusted value of mu to the instance variable self.mu
        self.mu = mu

        ### Set MT inputs
        # Set the affinity matrix structure
        allowed_structures = [
            "assortative",
            "disassortative",
            "core-periphery",
            "directed-biased",
        ]
        if structure not in allowed_structures:
            raise ValueError(
                f"The available structures for the affinity matrix w"
                f"are: {allowed_structures}."
            )
        self.structure = structure

        # Set eta parameter of the Dirichlet distribution
        if eta <= 0 and L1:
            raise ValueError("The Dirichlet parameter eta has to be positive!")
        self.eta = eta
        # Set alpha parameter of the Gamma distribution
        if ag <= 0 and not L1:
            raise ValueError("The Gamma parameter alpha has to be positive!")
        self.ag = ag
        # Set beta parameter of the Gamma distribution
        if bg <= 0 and not L1:
            raise ValueError("The Gamma parameter beta has to be positive!")
        self.bg = bg
        # Set u,v generation preference
        self.L1 = L1
        # Set correlation between u and v synthetically generated
        if (corr < 0) or (corr > 1):
            raise ValueError("The correlation parameter has to be in [0, 1]!")
        self.corr = corr
        # Set fraction of nodes with mixed membership
        if (over < 0) or (over > 1):
            raise ValueError("The overlapping parameter has to be in [0, 1]!")
        self.over = over

    def anomaly_network_PB(self, parameters=None):
        """
        Generate a directed, possibly weighted network by using the anomaly models Poisson-Poisson.

        Steps:
            1. Generate or load the latent variables Z_ij.
            2. Extract A_ij entries (network edges) from a Poisson (M_ij) distribution if Z_ij=0; from a Poisson (pi) distribution if Z_ij=1.

        Parameters
        ----------
        parameters
            Latent variables z, s, u, v, and w.

        Returns
        -------
        G : DiGraph
            DiGraph NetworkX object. Self-loops allowed.
        """
        ### Latent variables
        parameters = parameters if parameters else self._generate_lv()
        self.z, self.u, self.v, self.w = parameters

        # Network generation
        G = nx.DiGraph()
        for i in range(self.N):
            G.add_node(i)

        # Compute M_ij
        M = np.einsum("ik,jq->ijkq", self.u, self.v)
        M = np.einsum("ijkq,kq->ij", M, self.w)

        # Set c sparsity parameter
        c = brentq(
            eq_c, EPS, 20, args=(M, self.N, self.ExpM, self.rho_anomaly, self.mu)
        )

        self.w *= c

        # Build network
        A = self.rng.poisson(c * M)
        A[A > 0] = 1  # binarize the adjacency matrix
        np.fill_diagonal(A, 0)
        G0 = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        # weighted anomaly
        A[self.z.nonzero()] = self.rng.poisson(self.pi * self.z.count_nonzero())
        A[A > 0] = 1  # binarize the adjacency matrix
        np.fill_diagonal(A, 0)

        G = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        # Network post-processing

        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

        # Keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))
        # Define the nodes list again
        nodes = list(G.nodes())
        # Define the number of nodes
        self.N = len(nodes)
        # Redefine G0 as the subgraph defined by the nodes
        G0 = G0.subgraph(nodes)
        # Redefine the adjacency matrix A as the submatrix defined by the nodes
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")
        # Similar to A but this time from the subgraph G0
        A0 = nx.to_scipy_sparse_array(G0, nodelist=nodes, weight="weight")
        try:
            self.z = np.take(self.z, nodes, 1)
            self.z = np.take(self.z, nodes, 0)
        except IndexError:
            self.z = self.z[:, nodes]
            self.z = self.z[nodes]

        if self.u is not None:
            self.u = self.u[nodes]
            self.v = self.v[nodes]
        self.N = len(nodes)

        if self.verbose > 0:
            ave_deg = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
            logging.debug(
                "Number of nodes: %s \nNumber of edges: %s",
                G.number_of_nodes(),
                G.number_of_edges(),
            )
            logging.debug("Average degree (2E/N): %s", ave_deg)
            logging.debug(
                "rho_anomaly: %s",
                A[self.z.nonzero()].sum() / float(G.number_of_edges()),
            )

        if self.output_parameters:
            self._output_results(nodes)

        if self.output_adj:
            self._output_adjacency(G, outfile=self.outfile_adj)

        if self.verbose == 2:
            self._plot_A(A)
            self._plot_A(A0, title="A before anomaly")
            self._plot_A(self.z, title="Anomaly matrix Z")
            if M is not None:
                plot_M(M)

        return G, G0

    def _generate_lv(self):
        """
        Generate z, u, v, w latent variables.

        -------
        z
            Matrix NxN of models indicators (binary).

        u
            Matrix NxK of out-going membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        v
            Matrix NxK of in-coming membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        w
            Affinity matrix KxK. Possibly None if in pure SpringRank.
            Element (k,h) gives the density of edges going from the nodes
            of group k to nodes of group h.

        """
        # Generate z through binomial distribution

        if self.mu < 0:
            density = EPS
        else:
            density = self.mu
        z = sparse.random(
            self.N, self.N, density=density, data_rvs=np.ones, random_state=self.rng
        )
        upper_z = sparse.triu(z)
        z = upper_z + upper_z.T

        # Generate u, v for overlapping communities
        u, v = membership_vectors(
            self.rng,
            self.L1,
            self.eta,
            self.ag,
            self.bg,
            self.K,
            self.N,
            self.corr,
            self.over,
        )
        # Generate w
        w = affinity_matrix(
            structure=self.structure, N=self.N, K=self.K, avg_degree=self.avg_degree
        )

        return z, u, v, w

    def _output_results(self, nodes):
        """
        Output results in a compressed file.

        ----------
        nodes
            List of nodes IDs.

        """
        output_parameters = self.out_folder + "theta_" + self.label
        np.savez_compressed(
            output_parameters + ".npz",
            z=self.z.todense(),
            u=self.u,
            v=self.v,
            w=self.w,
            mu=self.mu,
            pi=self.pi,
            nodes=nodes,
        )

        logging.debug("Parameters saved in: %s.npz", output_parameters)
        logging.debug("To load: theta=np.load(filename), then e.g. theta['u']")

    def _plot_A(self, A, cmap="PuBuGn", title="Adjacency matrix"):
        """
        Plot the adjacency matrix produced by the generative algorithm.

        ----------
        A
            Sparse version of the NxN adjacency matrix associated with the graph.

        cmap
            Colormap used for the plot.

        title
            Title of the plot.

        """
        Ad = A.todense()
        _, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap=plt.get_cmap(cmap))
        ax.set_title(title, fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def _plot_Z(self, cmap="PuBuGn"):
        """
        Plot the anomaly matrix produced by the generative algorithm.

        ----------
        cmap
            Colormap used for the plot.

        """
        _, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.z, cmap=plt.get_cmap(cmap))
        ax.set_title("Anomaly matrix", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def _plot_M(self, M, cmap="PuBuGn", title="MT means matrix"):
        """
        Plot the M matrix produced by the generative algorithm.

        ----------
        M
            NxN M matrix associated with the graph. Contains all the means used
            for generating edges.

        cmap
            Colormap used for the plot.

        title
            Title of the plot.

        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(M, cmap=plt.get_cmap(cmap))
        ax.set_title(title, fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()
