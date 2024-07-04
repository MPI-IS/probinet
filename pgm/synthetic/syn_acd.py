"""
Class for generation and management of synthetic networks with anomalies
"""
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import scipy.sparse as sparse

from pgm.input.tools import flt
from pgm.synthetic.syn_dyncrep import eq_c, membership_vectors
from pgm.synthetic.syn_rep import affinity_matrix

EPS = 1e-12


class SyntNetAnomaly(object):

    def __init__(
        self,
        m=1,
        N=100,
        K=2,
        prng=10,
        avg_degree=4.0,
        rho_anomaly=0.1,
        structure="assortative",
        label=None,
        mu=None,
        pi=0.8,
        rho_node=0.9,
        gamma=0.5,
        eta=0.5,
        L1=False,
        ag=0.6,
        bg=1.0,
        corr=0.0,
        over=0.0,
        verbose=0,
        folder="../../data/input",
        output_parameters=False,
        output_adj=False,
        outfile_adj=None,
    ):

        # Set network size (node number)
        self.N = N
        # Set number of communities
        self.K = K
        # Set number of networks to be generated
        self.m = m
        # Set seed random number generator
        self.prng = prng
        # Set label (associated uniquely with the set of inputs)
        if label is not None:
            self.label = label
        else:
            self.label = ("_").join(
                [str(N), str(K), str(avg_degree), str(flt(rho_anomaly, d=2))]
            )
        # Initialize data folder path
        self.folder = folder
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

        if pi < 0 or pi > 1:
            raise ValueError("The Binomial parameter pi has to be in [0, 1]!")
        if pi == 1:
            pi = 1 - EPS
        if pi == 0:
            pi = EPS
        self.pi = pi

        if rho_anomaly < 0 or rho_anomaly > 1:
            raise ValueError("The rho anomaly has to be in [0, 1]!")

        self.ExpM = self.avg_degree * self.N * 0.5
        mu = (
            self.rho_anomaly
            * self.ExpM
            / ((1 - np.exp(-self.pi)) * (self.N**2 - self.N))
        )
        if mu == 1:
            mu = 1 - EPS
        if mu == 0:
            mu = EPS
        assert mu > 0.0 and mu < 1.0
        self.mu = mu

        ### Set MT inputs
        # Set the affinity matrix structure
        if structure not in [
            "assortative",
            "disassortative",
            "core-periphery",
            "directed-biased",
        ]:
            raise ValueError(
                "The available structures for the affinity matrix w "
                "are: assortative, disassortative, core-periphery "
                "and directed-biased!"
            )
        self.structure = structure

        # Set eta parameter of the  Dirichlet distribution
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
        """TODO: fix docstrings
        Generate a directed, possibly weighted network by using the anomaly model Poisson-Poisson
        Steps:
            1. Generate or load the latent variables Z_ij.
            2. Extract A_ij entries (network edges) from a Poisson (M_ij) distribution if Z_ij=0; from a Poisson (pi) distribution if Z_ij=1
        INPUT
        ----------
        parameters : object
                     Latent variables z, s, u, v and w.
        OUTPUT
        ----------
        G : Digraph
            DiGraph NetworkX object. Self-loops allowed.
        """

        # Set seed random number generator
        prng = np.random.RandomState(self.prng)

        ### Latent variables
        if parameters is None:
            # Generate latent variables
            self.z, self.u, self.v, self.w = self._generate_lv(prng)
        else:
            # Set latent variables
            self.z, self.u, self.v, self.w = parameters

        ### Network generation
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

        A = prng.poisson(c * M)
        A[A > 0] = 1  # binarize the adjacency matrix
        np.fill_diagonal(A, 0)
        G0 = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        # weighted anomaly
        A[self.z.nonzero()] = prng.poisson(self.pi * self.z.count_nonzero())
        A[A > 0] = 1  # binarize the adjacency matrix
        np.fill_diagonal(A, 0)

        G = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        # Network post-processing

        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight="weight")

        # Keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.N = len(nodes)

        G0 = G0.subgraph(nodes)
        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight="weight")  #
        A0 = nx.to_scipy_sparse_matrix(G0, nodelist=nodes, weight="weight")  #
        try:
            self.z = np.take(self.z, nodes, 1)
            self.z = np.take(self.z, nodes, 0)
        except:
            self.z = self.z[:, nodes]
            self.z = self.z[nodes]

        if self.u is not None:
            self.u = self.u[nodes]
            self.v = self.v[nodes]
        self.N = len(nodes)

        if self.verbose > 0:
            ave_deg = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
            logging.debug("Number of nodes: %s \nNumber of edges: %s", G.number_of_nodes(),
            G.number_of_edges())
            logging.debug("Average degree (2E/N): %s",ave_deg)
            print("rho_anomaly: %s", A[self.z.nonzero()].sum() / float(G.number_of_edges()))

        if self.output_parameters:
            self._output_results(nodes)

        if self.output_adj:
            self._output_adjacency(G, outfile=self.outfile_adj)

        if self.verbose == 2:
            self._plot_A(A)
            self._plot_A(A0, title="A before anomaly")
            self._plot_A(self.z, title="Anomaly matrix Z")
            if M is not None:
                self._plot_M(M)

        return G, G0

    def _generate_lv(self, prng=42):
        """#TODO: fix docstrings
        Generate z, u, v, w latent variables.
        INPUT
        ----------
        prng : int
               Seed for the random number generator.
        OUTPUT
        ----------
        z : Numpy array
            Matrix NxN of model indicators (binary).

        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        w : Numpy array
            Affinity matrix KxK. Possibly None if in pure SpringRank.
            Element (k,h) gives the density of edges going from the nodes
            of group k to nodes of group h.
        """
        # Generate z through binomial distribution

        if self.mu < 0:
            density = EPS
        else:
            density = self.mu
        z = sparse.random(self.N, self.N, density=density, data_rvs=np.ones)
        upper_z = sparse.triu(z)
        z = upper_z + upper_z.T

        # Generate u, v for overlapping communities
        u, v = membership_vectors(
            prng,
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
        w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree)
        return z, u, v, w

    def _output_results(self, nodes):
        """#TODO: fix docstrings
        Output results in a compressed file.
        INPUT
        ----------
        nodes : list
                List of nodes IDs.
        """
        output_parameters = self.folder + "theta_" + self.label + "_" + str(self.prng)
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


    def _output_adjacency(self, G, outfile=None):
        """#TODO: fix docstrings
        Output the adjacency matrix. Default format is space-separated .csv
        with 3 columns: node1 node2 weight
        INPUT
        ----------
        G: Digraph
           DiGraph NetworkX object.
        outfile: str
                 Name of the adjacency matrix.
        """
        if outfile is None:
            outfile = "syn_" + self.label + "_" + str(self.prng) + ".dat"

        edges = list(G.edges(data=True))
        try:
            data = [[u, v, d["weight"]] for u, v, d in edges]
        except:
            data = [[u, v, 1] for u, v, d in edges]

        df = pd.DataFrame(data, columns=["source", "target", "w"], index=None)
        df.to_csv(self.folder + outfile, index=False, sep=" ")
        logging.debug("Adjacency matrix saved in: %s", {self.folder + outfile})

    def _plot_A(self, A, cmap="PuBuGn", title="Adjacency matrix"):
        """TODO: fix docstrings
        Plot the adjacency matrix produced by the generative algorithm.
        INPUT
        ----------
        A : Scipy array
            Sparse version of the NxN adjacency matrix associated to the graph.
        cmap : Matplotlib object
            Colormap used for the plot.
        """
        Ad = A.todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap=plt.get_cmap(cmap))
        ax.set_title(title, fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def _plot_Z(self, cmap="PuBuGn"):
        """TODO: fix docstrings
        Plot the anomaly matrix produced by the generative algorithm.
        INPUT
        ----------
        cmap : Matplotlib object
            Colormap used for the plot.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.z, cmap=plt.get_cmap(cmap))
        ax.set_title("Anomaly matrix", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def _plot_M(self, M, cmap="PuBuGn", title="MT means matrix"):
        """TODO: fix docstrings
        Plot the M matrix produced by the generative algorithm. Each entry is the
        poisson mean associated to each couple of nodes of the graph.
        INPUT
        ----------
        M : Numpy array
            NxN M matrix associated to the graph. Contains all the means used
            for generating edges.
        cmap : Matplotlib object
            Colormap used for the plot.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(M, cmap=plt.get_cmap(cmap))
        ax.set_title(title, fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()
