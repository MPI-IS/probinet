"""
Class definition of the reciprocity generative model with the member functions required.
"""

import logging
import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import RandomState
import pandas as pd
import scipy
from scipy.sparse import coo_matrix

from ..input.tools import normalize_nonzero_membership
from ..model.constants import EPS_


class SyntheticDynCRep:
    """
    A class to generate a synthetic network using the DynCRep model.
    """

    def __init__(
        self,
        N: int,
        K: int,
        T: int = 1,
        eta: float = 0.0,  # reciprocity parameter
        L: int = 1,
        avg_degree: float = 5.0,
        prng: Union[int, np.random.RandomState] = 0,  # TODO: change this by seed
        verbose: int = 0,
        beta: float = 0.2,  # edge disappearance rate β(t)
        ag: float = 1.0,  # shape of gamma prior
        bg: float = 0.5,  # rate of gamma prior
        eta_dir: float = 0.5,
        L1: bool = True,  # u,v generation preference
        corr: float = 1.0,  # correlation between u and v synthetically generated
        over: float = 0.0,
        label: Optional[str] = None,
        end_file: str = ".dat",
        folder: str = "",
        structure: str = "assortative",
        output_parameters: bool = False,
        output_adj: bool = False,
        outfile_adj: Optional[str] = None,
        figsize: Tuple[int, int] = (7, 7),
        fontsize: int = 15,
        ExpM: Optional[np.ndarray] = None,
    ):
        """
        Initialize the CRepDyn class to generate a synthetic network using the DynCRep model.

        Parameters
        ----------
        N : int
            Number of nodes in the network.
        K : int
            Number of communities in the network.
        T : int, optional
            Number of time steps in the network.
        eta : float, optional
            Reciprocity parameter.
        L : int, optional
            Number of layers in the network.
        avg_degree : float, optional
            Average degree of nodes in the network.
        prng : Union[int, np.random.RandomState], optional # TODO: change this by seed
            Seed for random number generator.
        verbose : int, optional
            Verbosity level.
        beta : float, optional
            Parameter (beta) for Gamma.
        ag : float, optional
            Shape parameter of the gamma prior.
        bg : float, optional
            Rate parameter of the gamma prior.
        eta_dir : float, optional
            Parameter for Dirichlet.
        L1 : bool, optional
            Flag for parameter generation method. True for Dirichlet, False for Gamma.
        corr : float, optional
            Correlation between u and v synthetically generated.
        over : float, optional
            Fraction of nodes with mixed membership.
        label : str, optional
            Label for the model. This label is used as part of the filename when saving the
            output.
        end_file : str, optional
            File extension for output files.
        folder : str, optional
            Folder to save output files.
        structure : str, optional
            Structure of the network.
        output_parameters : bool, optional
            Whether to output parameters.
        output_adj : bool, optional
            Whether to output adjacency matrix.
        outfile_adj : str, optional
            File name for output adjacency matrix.
        figsize : Tuple[int, int], optional
            Size of the figures generated during the network creation process.
        fontsize : int, optional
            Font size of the figures generated during the network creation process.
        ExpM : np.ndarray, optional
            Expected number of edges in the network.
        """
        self.N = N
        self.K = K
        self.T = T
        self.L = L
        self.avg_degree = avg_degree
        self.prng = prng
        self.end_file = end_file
        # self.undirected = undirected # TODO: let Hadiseh know that this is not used
        self.folder = folder
        self.output_parameters = output_parameters
        self.output_adj = output_adj
        self.outfile_adj = outfile_adj
        self.figsize = figsize
        self.fontsize = fontsize

        if label is not None:
            self.label = label
        else:
            self.label = ("_").join(
                [str(N), str(K), str(avg_degree), str(T), str(eta), str(beta)]
            )
        self.structure = structure

        if ExpM is None:
            self.ExpM = self.avg_degree * self.N * 0.5
        else:
            self.ExpM = float(ExpM)

        # Set verbosity flag
        if verbose not in {0, 1, 2, 3}:
            raise ValueError(
                "The verbosity parameter can only assume values in {0,1,2,3}!"
            )
        self.verbose = verbose

        if eta < 0:
            raise ValueError("The parameter eta has to be positive!")
        self.eta = eta

        if beta < 0 or beta > 1:
            raise ValueError("The parameter beta has to be in [0, 1]!")
        if beta == 1:
            beta = 1 - EPS_
        if beta == 0:
            beta = EPS_
        self.beta = beta

        # Set MT inputs
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

        # Set alpha parameter of the Gamma distribution
        if ag <= 0 and not L1:
            raise ValueError("The Gamma parameter alpha has to be positive!")
        self.ag = ag
        # Set beta parameter of the Gamma distribution
        if bg <= 0 and not L1:
            raise ValueError("The Gamma parameter beta has to be positive!")
        self.bg = bg
        self.eta_dir = eta_dir
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

    def Exp_ija_matrix(self, u, v, w):
        """
        Compute the expected adjacency matrix for the DynCRep model.

        Parameters
        ----------
        u : np.ndarray
            Outgoing membership matrix.
        v : np.ndarray
            Incoming membership matrix.
        w : np.ndarray
            Affinity tensor.

        Returns
        -------
        Exp_ija : np.ndarray
            Expected adjacency matrix.
        """
        # Compute the product of the outgoing membership matrix and the affinity tensor
        Exp_ija = np.einsum("ik,kq->iq", u, w)

        # Compute the product of the result and the incoming membership matrix
        Exp_ija = np.einsum("iq,jq->ij", Exp_ija, v)

        return Exp_ija

    def generate_net(self, parameters=None):
        """
        Generate a directed, possibly weighted network by using DynCRep.

        Steps:
            1. Generate a network A[0].
            2. Extract A[t] entries (network edges) using transition probabilities.

        Parameters
        ----------
        parameters : dict, optional
            Latent variables eta, beta, u, v and w.

        Returns
        -------
        G : networkx.classes.digraph.DiGraph
            NetworkX DiGraph object. Self-loops allowed.
        """

        # Set seed random number generator
        prng = np.random.RandomState(self.prng)

        # Latent variables
        if parameters is None:
            # Generate latent variables
            self.u, self.v, self.w = self._generate_lv(prng)
        else:
            # Set latent variables
            self.u, self.v, self.w = parameters

        # Network generation
        G = [nx.DiGraph() for t in range(self.T + 1)]
        for t in range(self.T + 1):
            for i in range(self.N):
                G[t].add_node(i)

        # Compute M_ij
        M = self.Exp_ija_matrix(self.u, self.v, self.w)
        np.fill_diagonal(M, 0)

        # Set c sparsity parameter
        Exp_M_inferred = M.sum()
        c = self.ExpM / Exp_M_inferred
        # For t = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    A_ij = prng.poisson(c * M[i, j], 1)[0]
                    if A_ij > 0:
                        G[0].add_edge(i, j, weight=1)  # binarized

        # For t > 0
        for t in range(self.T):
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:

                        if G[t].has_edge(
                            j, i
                        ):  # reciprocal edge determines Poisson rate
                            lambda_ij = c * M[i, j] + self.eta
                        else:
                            lambda_ij = c * M[i, j]

                        if G[t].has_edge(
                            i, j
                        ):  # the edge at previous time step: determines the transition rate
                            q = 1 - self.beta
                        else:
                            q = self.beta * lambda_ij
                        r = prng.rand()
                        if r <= q:
                            G[t + 1].add_edge(i, j, weight=1)  # binarized

        # Network post-processing
        nodes = list(G[0].nodes())
        assert len(nodes) == self.N
        A = [
            nx.to_scipy_sparse_array(G[t], nodelist=nodes, weight="weight")
            for t in range(len(G))
        ]

        # Keep largest connected component
        A_sum = A[0].copy()
        for t in range(1, len(A)):
            A_sum += A[t]
        G_sum = nx.from_scipy_sparse_array(A_sum, create_using=nx.DiGraph)
        Gc = max(nx.weakly_connected_components(G_sum), key=len)
        nodes_to_remove = set(G_sum.nodes()).difference(Gc)
        G_sum.remove_nodes_from(list(nodes_to_remove))

        if self.output_adj:
            self._output_adjacency(
                nodes,
                A_sum,
                A,
                nodes_to_keep=list(G_sum.nodes()),
                outfile=self.outfile_adj,
            )

        nodes = list(G_sum.nodes())

        for t in range(len(G)):
            G[t].remove_nodes_from(list(nodes_to_remove))

        if self.u is not None:
            self.u = self.u[nodes]
            self.v = self.v[nodes]
        self.N = len(nodes)

        if self.verbose > 0:
            if len(nodes_to_remove) > 0:
                logging.debug(
                    "Removed %s nodes, because not part of the largest connected component",
                    len(nodes_to_remove),
                )

        if self.verbose > 0:
            for t in range(len(G)):
                logging.debug("-" * 30)
                logging.debug("t=%s", t)
                ave_w_deg = np.round(
                    2 * G[t].number_of_edges() / float(G[t].number_of_nodes()), 3
                )
                logging.debug(
                    "Number of nodes: %s \nNumber of edges: %s",
                    G[t].number_of_nodes(),
                    G[t].number_of_edges(),
                )
                logging.debug("Average degree (2E/N): %s", ave_w_deg)
                logging.debug("Reciprocity at t: %s", nx.reciprocity(G[t]))
                logging.debug("-" * 30)
            self.check_reciprocity_tm1(A)  # A_sum was passed originally too

        if self.output_parameters:
            self._output_results(nodes)

        if self.verbose >= 2:
            self._plot_A(A, figsize=self.figsize, fontsize=self.fontsize)
            if self.verbose == 3:
                if M is not None:
                    self._plot_M(M, figsize=self.figsize, fontsize=self.fontsize)

        return G

    def _generate_lv(self, prng: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate u, v, w latent variables.

        Parameters
        ----------
        prng : int, optional
            Seed for the random number generator. Default is 42.

        Returns
        ----------
        u : np.ndarray
            Matrix NxK of out-going membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        v : np.ndarray
            Matrix NxK of in-coming membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        w : np.ndarray
            Affinity matrix KxK. Possibly None if in pure SpringRank.
            Element (k,h) gives the density of edges going from the nodes
            of group k to nodes of group h.
        """

        # Generate u, v for overlapping communities
        u, v = membership_vectors(
            prng,
            self.L1,
            self.eta_dir,
            self.ag,
            self.bg,
            self.K,
            self.N,
            self.corr,
            self.over,
        )
        # Generate w
        w = affinity_matrix_dyncrep(self.structure, self.N, self.K, self.avg_degree)
        return u, v, w

    def _build_multilayer_edgelist(
        self,
        A_tot: coo_matrix,
        A: List[coo_matrix],
        nodes_to_keep: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Build a multilayer edge list from the given adjacency matrices.

        This function converts the total adjacency matrix to a coordinate format and initializes a dictionary to store the source and target nodes.
        It then loops over each adjacency matrix in A and adds the weights of the edges at each time t to the dictionary.

        Parameters
        ----------
        A_tot : scipy.sparse.coo_matrix
            The total adjacency matrix.
        A : List[scipy.sparse.coo_matrix]
            List of adjacency matrices for each layer.
        nodes_to_keep : List[int], optional
            List of node IDs to keep. If not provided, all nodes are kept.

        Returns
        -------
        df_res : pd.DataFrame
            DataFrame containing the multilayer edge list.
        """
        # Convert the total adjacency matrix to a coordinate format
        A_coo = A_tot.tocoo()

        # Initialize a dictionary to store the source and target nodes
        data_dict = {"source": A_coo.row, "target": A_coo.col}

        # Loop over each adjacency matrix in A
        for t in range(len(A)):
            # Add the weights of the edges at time t to the dictionary
            data_dict["weight_t" + str(t)] = np.squeeze(
                np.asarray(A[t][A_tot.nonzero()])
            )

        # Convert the dictionary to a DataFrame
        df_res = pd.DataFrame(data_dict)

        # If nodes_to_keep is not None, filter the DataFrame to only include these nodes
        if nodes_to_keep is not None:
            df_res = df_res[
                df_res.source.isin(nodes_to_keep) & df_res.target.isin(nodes_to_keep)
            ]

        # Get the list of unique nodes
        nodes = list(set(df_res.source).union(set(df_res.target)))

        # Create a dictionary mapping node IDs to nodes
        id2node = {i: n for i, n in enumerate(nodes)}

        # Replace the source and target node IDs with the actual nodes
        df_res["source"] = df_res.source.map(id2node)
        df_res["target"] = df_res.target.map(id2node)

        # Return the resulting DataFrame
        return df_res

    def _output_results(self, nodes: List[int]) -> None:
        """
        Output results in a compressed file.

        Parameters
        ----------
        nodes : List[int]
            List of node IDs.
        """
        output_parameters = self.folder + "theta_" + self.label + "_" + str(self.prng)
        np.savez_compressed(
            output_parameters + ".npz",
            u=self.u,
            v=self.v,
            w=self.w,
            eta=self.eta,
            beta=self.beta,
            nodes=nodes,
        )

        logging.debug("Parameters saved in: %s.npz", output_parameters)
        logging.debug("To load: theta=np.load(filename), then e.g. theta['u']")

    def _output_adjacency(
        self,
        nodes: List[int],
        A_tot: scipy.sparse.coo_matrix,
        A: List[scipy.sparse.coo_matrix],
        nodes_to_keep: Optional[List[int]] = None,
        outfile: Optional[str] = None,
    ) -> None:
        """
        Output the adjacency matrix. The default format is a space-separated .csv file with 3 columns: node1, node2, and weight.

        Parameters
        ----------
        nodes : List[int]
            List of node IDs.
        A_tot : scipy.sparse.coo_matrix
            The total adjacency matrix.
        A : List[scipy.sparse.coo_matrix]
            List of adjacency matrices for each layer.
        nodes_to_keep : List[int], optional
            List of node IDs to keep. If not provided, all nodes are kept.
        outfile : str, optional
            Name of the output file for the adjacency matrix. If not provided, a default name is used.
        """
        if outfile is None:
            outfile = "syn_" + self.label + "_" + str(self.prng) + ".dat"

        df = self._build_multilayer_edgelist(A_tot, A, nodes_to_keep=nodes_to_keep)
        df.to_csv(self.folder + outfile, index=False, sep=" ")
        logging.debug("Adjacency matrix saved in: %s", self.folder + outfile)

    def _plot_A(
        self,
        A: np.ndarray,
        cmap: str = "PuBuGn",
        figsize: Tuple[int, int] = (7, 7),
        fontsize: int = 15,
    ) -> None:
        """
        Plot the adjacency matrix produced by the generative algorithm.

        Parameters
        ----------
        A : np.ndarray
            Sparse version of the NxN adjacency matrix associated to the graph.
        cmap : str, optional
            Colormap used for the plot.
        figsize : Tuple[int, int], optional
            Size of the figure to be plotted.
        fontsize : int, optional
            Font size to be used in the plot title.
        """
        for i in range(len(A)):
            Ad = A[i].todense()
            _, ax = plt.subplots(figsize=figsize)
            ax.matshow(Ad, cmap=plt.get_cmap(cmap))
            ax.set_title(f"Adjacency matrix at time {i}", fontsize=fontsize)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)

    def _plot_M(
        self,
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

    def check_reciprocity_tm1(self, A: List[coo_matrix]) -> None:
        """
        Check the reciprocity of the adjacency matrices at time t and t-1.

        Parameters
        ----------
        A : List[scipy.sparse.coo_matrix]
            List of adjacency matrices for each time step.
        """
        if len(A) > 1:
            logging.debug("Compare current and previous reciprocity.")
        # Loop over each adjacency matrix in A, starting from the second one
        for t in range(1, len(A)):
            # Get the indices of the non-zero elements in the adjacency matrix at time t
            ref_subs = A[t].nonzero()

            # Get the non-zero elements in the transposed adjacency matrix at time t
            M_t_T = A[t].transpose()[ref_subs]

            # Get the non-zero elements in the transposed adjacency matrix at time t-1
            M_tm1_T = A[t - 1].transpose()[ref_subs]

            # Get the number of non-zero elements in the adjacency matrix at time t
            nnz = float(A[t].count_nonzero())

            # Log the number of non-zero elements in the adjacency matrix at time t,
            # the fraction of non-zero elements in the transposed adjacency matrix at time t,
            # and the fraction of non-zero elements in the transposed adjacency matrix at time t-1
            logging.debug("Time step: %s", t)
            logging.debug(
                "Number of non-zero elements in the adjacency matrix at time t: %s", nnz
            )
            logging.debug(
                "Fraction of non-zero elements in the transposed adjacency matrix at time t: %s",
                M_t_T.nonzero()[0].shape[0] / nnz,
            )
            logging.debug(
                "Fraction of non-zero elements in the transposed adjacency matrix at time t-1: %s",
                M_tm1_T.nonzero()[0].shape[0] / nnz,
            )


def membership_vectors(
    prng: RandomState = RandomState(10),
    # TODO: change this to use random seeds instead of prng
    L1: bool = False,
    eta_dir: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1,
    K: int = 2,
    N: int = 100,
    corr: float = 0.0,
    over: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the NxK membership vectors u, v using a Dirichlet or a Gamma distribution.

    Parameters
    ----------
    prng : numpy.random.RandomState
        Random number generator container.
    L1 : bool
        Flag for parameter generation method. True for Dirichlet, False for Gamma.
    eta_dir : float
        Parameter for Dirichlet.
    alpha : float
        Parameter (alpha) for Gamma.
    beta : float
        Parameter (beta) for Gamma.
    K : int
        Number of communities.
    N : int
        Number of nodes.
    corr : float
        Correlation between u and v synthetically generated.
    over : float
        Fraction of nodes with mixed membership.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure Multitensor.
            With unitary L1 norm computed row-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure Multitensor.
            With unitary L1 norm computed row-wise.
    """
    # Generate equal-size unmixed group membership
    size = int(N / K)
    u = np.zeros((N, K))
    v = np.zeros((N, K))
    for i in range(N):
        q = int(math.floor(float(i) / float(size)))
        if q == K:
            u[i:, K - 1] = 1.0
            v[i:, K - 1] = 1.0
        else:
            for j in range(q * size, q * size + size):
                u[j, q] = 1.0
                v[j, q] = 1.0
    # Generate mixed communities if requested
    if over != 0.0:
        overlapping = int(N * over)  # number of nodes belonging to more communities
        ind_over = np.random.randint(len(u), size=overlapping)
        if L1:
            u[ind_over] = prng.dirichlet(
                eta_dir * np.ones(K), overlapping
            )  # TODO: Ask Hadiseh  why eta is not defined. Answer: its probably eta_dir
            v[ind_over] = corr * u[ind_over] + (1.0 - corr) * prng.dirichlet(
                eta_dir * np.ones(K), overlapping  # pylint: disable=undefined-variable
            )
            if corr == 1.0:
                assert np.allclose(u, v)
            if corr > 0:
                v = normalize_nonzero_membership(v)
        else:
            u[ind_over] = prng.gamma(alpha, 1.0 / beta, size=(N, K))
            v[ind_over] = corr * u[ind_over] + (1.0 - corr) * prng.gamma(
                alpha, 1.0 / beta, size=(overlapping, K)
            )
            u = normalize_nonzero_membership(u)
            v = normalize_nonzero_membership(v)
    return u, v


def affinity_matrix_dyncrep(
    structure: str = "assortative",
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
    structure : str
                Structure of the network. Default is 'assortative'.
    N : int
        Number of nodes. Default is 100.
    K : int
        Number of communities. Default is 2.
    avg_degree : float
        Average degree. Default is 4.0.
    a : float
        Parameter for secondary probabilities. Default is 0.1.
    b : float
        Parameter for secondary probabilities. Default is 0.3.

    Returns
    -------
    p : np.ndarray
        Array with probabilities between and within groups. Element (k,h)
        gives the density of edges going from the nodes of group k to nodes of group h.
    """

    b *= a
    p1 = avg_degree * K / N

    if structure == "assortative":
        p = p1 * a * np.ones((K, K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == "disassortative":
        p = p1 * np.ones((K, K))  # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    elif structure == "core-periphery":
        p = p1 * np.ones((K, K))
        np.fill_diagonal(np.fliplr(p), a * p1)
        p[1, 1] = b * p1

    elif structure == "directed-biased":
        p = a * p1 * np.ones((K, K))
        p[0, 1] = p1
        p[1, 0] = b * p1

    return p


def eq_c(c: float, M: np.ndarray, N: int, E: int, rho_a: float, mu: float) -> float:
    """
    Compute the value of a function used to find the value of the sparsity parameter 'c'.

    Parameters
    ----------
    c : float
        The sparsity parameter.
    M : np.ndarray
        The matrix representing the expected number of edges between each pair of nodes.
    N : int
        The number of nodes in the network.
    E : int
        The expected total number of edges in the network.
    rho_a : float
        The expected proportion of reciprocal edges in the network.
    mu : float
        The proportion of reciprocal edges in the Erdos-Renyi network.

    Returns
    -------
    float
        The value of the function for the given parameters.
    """
    return np.sum(np.exp(-c * M)) - (N**2 - N) + E * (1 - rho_a) / (1 - mu)
