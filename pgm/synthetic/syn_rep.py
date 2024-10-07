"""
Class definition of the reciprocity generative model with the member functions required.
It builds a directed, possibly weighted, network.
"""

import logging
import math
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import tril, triu

from ..input.stats import reciprocal_edges
from ..input.tools import (
    Exp_ija_matrix, log_and_raise_error, normalize_nonzero_membership, transpose_ij2)

# TODO: add type hints into a separate script

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GM_reciprocity:
    """
    A class to generate a directed, possibly weighted, network with reciprocity.
    """

    def __init__(
        self,
        N: int,
        K: int,
        eta: float = 0.5,
        k: float = 3,
        over: float = 0.0,
        corr: float = 0.0,
        seed: int = 0,
        alpha: float = 0.1,
        ag: float = 0.1,
        beta: float = 0.1,
        Normalization: int = 0,
        structure: str = "assortative",
        end_file: str = None,
        out_folder: str = None,
        output_parameters: bool = False,
        output_adj: bool = False,
        outfile_adj: str = None,
        ExpM: Optional[float] = None,
    ):
        self.N = N  # number of nodes
        self.K = K  # number of communities
        self.k = k  # average degree
        self.seed = seed  # random seed
        self.alpha = alpha  # parameter of the Dirichlet distribution
        self.ag = ag  # alpha parameter of the Gamma distribution
        self.beta = beta  # beta parameter of the Gamma distribution
        self.end_file = end_file  # output file suffix
        self.out_folder = out_folder  # path for storing the output
        self.output_parameters = output_parameters  # flag for storing the parameters
        self.output_adj = output_adj  # flag for storing the generated adjacency matrix
        self.outfile_adj = outfile_adj  # name for saving the adjacency matrix
        if (eta < 0) or (eta >= 1):  # reciprocity coefficient
            log_and_raise_error(
                ValueError, "The reciprocity coefficient eta has to be in [0, 1)!"
            )
        self.eta = eta
        if ExpM is None:  # expected number of edges
            self.ExpM = int(self.N * self.k / 2.0)
        else:
            self.ExpM = int(ExpM)
            self.k = 2 * self.ExpM / float(self.N)
        if (over < 0) or (over > 1):  # fraction of nodes with mixed membership
            log_and_raise_error(
                ValueError, "The overlapping parameter has to be in [0, 1]!"
            )
        self.over = over
        if (corr < 0) or (
            corr > 1
        ):  # correlation between u and v synthetically generated
            log_and_raise_error(
                ValueError, "The correlation parameter corr has to be in [0, 1]!"
            )

        self.corr = corr
        if Normalization not in {
            0,
            1,
        }:  # indicator for choosing how to generate the latent variables
            message = (
                "The Normalization parameter can be either 0 or 1! It is used as an "
                "indicator for generating the membership matrices u and v from a Dirichlet or a Gamma "
                "distribution, respectively. It is used when there is overlapping."
            )
            log_and_raise_error(ValueError, message)
        self.Normalization = Normalization
        if structure not in {
            "assortative",
            "disassortative",
        }:  # structure of the affinity matrix W
            message = (
                "The structure of the affinity matrix w can be either assortative or "
                "disassortative!"
            )
            log_and_raise_error(ValueError, message)
        self.structure = structure

    def reciprocity_planted_network(
        self,
        parameters: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None,
    ) -> Tuple[
        nx.MultiDiGraph, np.ndarray
    ]:  # this could be called CRep (synthetic.CRep)
        """
        Generate a directed, possibly weighted network by using the reciprocity generative model.
        Can be used to generate benchmarks for networks with reciprocity.

        Steps:
            1. Generate the latent variables.
            2. Extract A_ij entries (network edges) from a Poisson distribution;
               its mean depends on the latent variables.

        Parameters
        ----------
        parameters: Tuple[np.ndarray, np.ndarray, np.ndarray, float], optional
                    Latent variables u, v, w, and eta.

        Returns
        -------
        G: MultiDiGraph
           MultiDiGraph NetworkX object.
        A: np.ndarray
           The adjacency matrix of the generated network.
        """

        # Create a random number generator with a specific seed
        prng = np.random.RandomState(self.seed)

        # Check if parameters are provided
        if parameters is not None:
            # If parameters are provided, set u, v, w, and eta to the provided values
            self.u, self.v, self.w, self.eta = parameters
        else:
            # If parameters are not provided, initialize u, v, and w
            # Calculate the size of each community
            size = int(self.N / self.K)

            # Initialize u and v as zero matrices
            self.u = np.zeros((self.N, self.K))
            self.v = np.zeros((self.N, self.K))

            # Loop over all nodes
            for i in range(self.N):
                # Calculate the community index for the current node
                q = int(math.floor(float(i) / float(size)))

                # If the community index is equal to the number of communities
                if q == self.K:
                    # Assign the last community to the remaining nodes
                    self.u[i:, self.K - 1] = 1.0
                    self.v[i:, self.K - 1] = 1.0
                else:
                    # Assign the current community to the nodes in the current range
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.0
                        self.v[j, q] = 1.0

            # Generate the affinity matrix w
            self.w = affinity_matrix(
                structure=self.structure, N=self.N, K=self.K, a=0.1, b=0.3
            )

            # Check if there is overlapping in the communities
            if self.over != 0.0:
                # Calculate the number of nodes belonging to more communities
                overlapping = int(self.N * self.over)
                # Randomly select 'overlapping' number of nodes
                ind_over = np.random.randint(len(self.u), size=overlapping)

                # Check the normalization method
                if self.Normalization == 0:
                    # If Normalization is 0, generate u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(
                        self.alpha * np.ones(self.K), overlapping
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.dirichlet(self.alpha * np.ones(self.K), overlapping)

                    # If correlation is 1, ensure u and v are close
                    if self.corr == 1.0:
                        assert np.allclose(self.u, self.v)

                    # If correlation is greater than 0, normalize v
                    if self.corr > 0:
                        self.v = normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1.0 / self.beta, size=(overlapping, self.K)
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.gamma(self.ag, 1.0 / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = normalize_nonzero_membership(self.u)
                    self.v = normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)

        # Compute the constant to enforce sparsity in the network
        c = (self.ExpM * (1.0 - self.eta)) / M0.sum()

        # Compute the expected number of edges between each pair of nodes considering reciprocity
        MM = (M0 + self.eta * transpose_ij2(M0)) / (
            1.0 - self.eta * self.eta
        )  # whose elements are m_{ij}
        Mt = transpose_ij2(MM)
        MM0 = M0.copy()  # to be not influenced by c_lambda

        # Adjust the affinity matrix w and the expected number of edges M0 by the constant c
        if parameters is None:
            self.w *= c  # only w is impact by that, u and v have a constraint,
            # their sum over k should sum to 1
        M0 *= c
        M0t = transpose_ij2(M0)  # whose elements are lambda0_{ji}

        # Compute the expected number of edges between each pair of nodes considering reciprocity
        M = (M0 + self.eta * M0t) / (
            1.0 - self.eta * self.eta
        )  # whose elements are m_{ij}
        np.fill_diagonal(M, 0)

        # Compute the expected reciprocity in the network
        Exp_r = self.eta + ((MM0 * Mt + self.eta * Mt**2).sum() / MM.sum())

        # Generate the network G and the adjacency matrix A using the latent variables
        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        counter, totM = 0, 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = prng.rand(1)[0]
                if r < 0.5:
                    # Draw the number of edges from node i to node j from a Poisson distribution
                    A_ij = prng.poisson(M[i, j], 1)[
                        0
                    ]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                    # Compute the expected number of edges from node j to node i considering
                    # reciprocity
                    lambda_ji = M0[j, i] + self.eta * A_ij
                    # Draw the number of edges from node j to node i from a Poisson distribution
                    A_ji = prng.poisson(lambda_ji, 1)[
                        0
                    ]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                else:
                    # Draw the number of edges from node j to node i from a Poisson distribution
                    A_ji = prng.poisson(M[j, i], 1)[
                        0
                    ]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                    # Compute the expected number of edges from node i to node j considering
                    # reciprocity
                    lambda_ij = M0[i, j] + self.eta * A_ji
                    # Draw the number of edges from node i to node j from a Poisson distribution
                    A_ij = prng.poisson(lambda_ij, 1)[
                        0
                    ]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                counter += 1
                totM += A_ij + A_ji

        # Keep only the largest connected component of the network
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        # Update the list of nodes and the number of nodes
        nodes = list(G.nodes())
        self.u = self.u[nodes]
        self.v = self.v[nodes]
        self.N = len(nodes)

        # Convert the network to a sparse adjacency matrix
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

        # Compute the average degree and the average weighted degree in the network
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Compute the weighted reciprocity
        rw = np.multiply(A, A.T).sum() / A.sum()

        logging.info(
            "Number of links in the upper triangular matrix: %s", triu(A, k=1).nnz
        )
        logging.info(
            "Number of links in the lower triangular matrix: %s", tril(A, k=-1).nnz
        )
        logging.info(
            "Sum of weights in the upper triangular matrix: %.3f",
            triu(A, k=1).sum(),
        )
        logging.info(
            "Sum of weights in the lower triangular matrix: %.3f",
            tril(A, k=-1).sum(),
        )
        logging.info("Number of possible unordered pairs: %s", counter)
        logging.info(
            "Removed %s nodes, because not part of the largest connected component",
            len(nodes_to_remove),
        )
        logging.info("Number of nodes: %s", G.number_of_nodes())
        logging.info("Number of edges: %s", G.number_of_edges())
        logging.info("Average degree (2E/N): %s", Sparsity_cof)
        logging.info("Average weighted degree (2M/N): %s", ave_w_deg)
        logging.info("Expected reciprocity: %.3f", Exp_r)
        logging.info("Reciprocity (networkX) = %.3f", nx.reciprocity(G))
        logging.info("Reciprocity (considering the weights of the edges) = %.3f", rw)

        # Output the parameters of the network if output_parameters is True
        if self.output_parameters:
            self.output_results(nodes)

        # Output the adjacency matrix of the network if output_adj is True
        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G, A

    def planted_network_cond_independent(
        self, parameters: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> Tuple[nx.MultiDiGraph, np.ndarray]:
        """
        Generate a directed, possibly weighted network without using reciprocity.
        It uses conditionally independent A_ij from a Poisson | (u,v,w).

        Parameters
        ----------
        parameters: Tuple[np.ndarray, np.ndarray, np.ndarray], optional
                    Latent variables u, v, and w.

        Returns
        -------
        G: MultiDigraph
           MultiDiGraph NetworkX object.
        A: np.ndarray
            The adjacency matrix of the generated network.
        """

        # Create a random number generator with a specific seed
        prng = np.random.RandomState(self.seed)

        # Set latent variables u,v,w
        if parameters is not None:
            # If parameters are provided, set u, v, w to the provided values
            self.u, self.v, self.w = parameters
        else:
            # If parameters are not provided, initialize u, v, and w
            # Calculate the size of each community
            size = int(self.N / self.K)

            # Initialize u and v as zero matrices
            self.u = np.zeros((self.N, self.K))
            self.v = np.zeros((self.N, self.K))

            # Loop over all nodes
            for i in range(self.N):
                # Calculate the community index for the current node
                q = int(math.floor(float(i) / float(size)))

                # If the community index is equal to the number of communities
                if q == self.K:
                    # Assign the last community to the remaining nodes
                    self.u[i:, self.K - 1] = 1.0
                    self.v[i:, self.K - 1] = 1.0
                else:
                    # Assign the current community to the nodes in the current range
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.0
                        self.v[j, q] = 1.0

            # Generate the affinity matrix w
            self.w = affinity_matrix(
                structure=self.structure, N=self.N, K=self.K, a=0.1, b=0.3
            )

            # Check if there is overlapping in the communities
            if self.over != 0.0:
                # Calculate the number of nodes belonging to more communities
                overlapping = int(self.N * self.over)
                # Randomly select 'overlapping' number of nodes
                ind_over = np.random.randint(len(self.u), size=overlapping)

                # Check the normalization method
                if self.Normalization == 0:
                    # If Normalization is 0, generate u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(
                        self.alpha * np.ones(self.K), overlapping
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.dirichlet(self.alpha * np.ones(self.K), overlapping)

                    # If correlation is 1, ensure u and v are close
                    if self.corr == 1.0:
                        assert np.allclose(self.u, self.v)

                    # If correlation is greater than 0, normalize v
                    if self.corr > 0:
                        self.v = normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1.0 / self.beta, size=(overlapping, self.K)
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.gamma(self.ag, 1.0 / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = normalize_nonzero_membership(self.u)
                    self.v = normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)
        M0t = transpose_ij2(M0)  # whose elements are lambda0_{ji}

        # Compute the expected reciprocity in the network
        rw = (M0 * M0t).sum() / M0.sum()  # expected reciprocity

        # Compute the constant to enforce sparsity in the network
        c = self.ExpM / float(M0.sum())  # constant to enforce sparsity

        # Adjust the affinity matrix w and the expected number of edges M0 by the constant c
        if parameters is None:
            self.w *= c  # only w is impact by that, u and v have a constraint, their sum over k should sum to 1

        # Generate network G (and adjacency matrix A) using the latent variable,
        # with the generative model (A_ij) ~ P(A_ij|u,v,w)
        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        totM = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:  # no self-loops
                    # draw A_ij from P(A_ij) = Poisson(c*m_ij)
                    A_ij = prng.poisson(c * M0[i, j], 1)[0]
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                    totM += A_ij

        nodes = list(G.nodes())

        # keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        # Update the list of nodes and the number of nodes
        nodes = list(G.nodes())
        self.u = self.u[nodes]
        self.v = self.v[nodes]
        self.N = len(nodes)

        # Convert the network to a sparse adjacency matrix
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

        # Calculate the average degree and the average weighted degree in the graph
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Calculate the proportion of bi-directional edges over the unordered pairs of nodes
        reciprocity_c = np.round(reciprocal_edges(G), 3)

        logging.info(
            "Number of links in the upper triangular matrix: %s", triu(A, k=1).nnz
        )
        logging.info(
            "Number of links in the lower triangular matrix: %s", tril(A, k=-1).nnz
        )
        logging.info(
            "Sum of weights in the upper triangular matrix: %s",
            np.round(triu(A, k=1).sum(), 2),
        )
        logging.info(
            "Sum of weights in the lower triangular matrix: %s",
            np.round(tril(A, k=-1).sum(), 2),
        )
        logging.info(
            "Removed %s nodes, because not part of the largest connected component",
            len(nodes_to_remove),
        )
        logging.info("Number of nodes: %s", G.number_of_nodes())
        logging.info("Number of edges: %s", G.number_of_edges())
        logging.info("Average degree (2E/N): %s", Sparsity_cof)
        logging.info("Average weighted degree (2M/N): %s", ave_w_deg)
        logging.info("Expected reciprocity: %.3f", rw)
        logging.info(
            "Reciprocity (intended as the proportion of bi-directional edges over the "
            "unordered pairs): %s",
            reciprocity_c,
        )

        # Output the parameters of the network if output_parameters is True
        if self.output_parameters:
            self.output_results(nodes)

        # Output the adjacency matrix of the network if output_adj is True
        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G, A

    def planted_network_reciprocity_only(
        self, p: Optional[float] = None
    ) -> Tuple[nx.MultiDiGraph, np.ndarray]:
        """
        Generate a directed, possibly weighted network using only reciprocity.
        One of the directed-edges is generated with probability p, the other with eta*A_ji,
        i.e. as in Erdos-Renyi reciprocity.

        Parameters
        ----------
        p: float, optional
           Probability to generate one of the directed-edge.

        Returns
        -------
        G: MultiDigraph
           MultiDiGraph NetworkX object.
        A: np.ndarray
            The adjacency matrix of the generated network.
        """

        # Create a random number generator with a specific seed
        prng = np.random.RandomState(self.seed)

        # If p is not provided, calculate it based on eta, k, and N
        if p is None:
            p = (1.0 - self.eta) * self.k * 0.5 / (self.N - 1.0)

        # Initialize a directed graph G
        G = nx.MultiDiGraph()
        for i in range(self.N):
            G.add_node(i)

        # Initialize total weight of the graph
        totM = 0

        # Generate edges for the graph
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Draw two random numbers from Poisson distribution
                A0 = prng.poisson(p, 1)[0]
                A1 = prng.poisson(p + A0, 1)[0]
                r = prng.rand(1)[0]
                # Add edges to the graph based on the drawn numbers
                if r < 0.5:
                    if A0 > 0:
                        G.add_edge(i, j, weight=A0)
                    if A1 > 0:
                        G.add_edge(j, i, weight=A1)
                else:
                    if A0 > 0:
                        G.add_edge(j, i, weight=A0)
                    if A1 > 0:
                        G.add_edge(i, j, weight=A1)
                # Update total weight of the graph
                totM += A0 + A1

        # Keep only the largest connected component of the graph
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        # Update the list of nodes and the number of nodes
        nodes = list(G.nodes())
        self.N = len(nodes)

        # Convert the graph to a sparse adjacency matrix
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

        # Calculate the average degree and the average weighted degree in the graph
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Calculate the proportion of bi-directional edges over the unordered pairs of nodes
        reciprocity_c = np.round(reciprocal_edges(G), 3)

        logging.info(
            "Number of links in the upper triangular matrix: %s", triu(A, k=1).nnz
        )
        logging.info(
            "Number of links in the lower triangular matrix: %s", tril(A, k=-1).nnz
        )
        logging.info(
            "Sum of weights in the upper triangular matrix: %s",
            np.round(triu(A, k=1).sum(), 2),
        )
        logging.info(
            "Sum of weights in the lower triangular matrix: %s",
            np.round(tril(A, k=-1).sum(), 2),
        )
        logging.info(
            "Removed %s nodes, because not part of the largest connected component",
            len(nodes_to_remove),
        )
        logging.info("Number of nodes: %s", G.number_of_nodes())
        logging.info("Number of edges: %s", G.number_of_edges())
        logging.info("Average degree (2E/N): %s", Sparsity_cof)
        logging.info("Average weighted degree (2M/N): %s", ave_w_deg)
        logging.info(
            "Reciprocity (intended as the proportion of bi-directional edges over the "
            "unordered pairs): %s",
            reciprocity_c,
        )

        # Output the adjacency matrix of the graph if output_adj is True
        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G, A

    def output_results(self, nodes: List[int]) -> None:
        """
        Output results in a compressed file.

        Parameters
        ----------
        nodes : List[int]
                List of nodes IDs.
        """

        output_parameters = (
            self.out_folder + "theta_gt" + str(self.seed) + self.end_file
        )
        np.savez_compressed(
            output_parameters + ".npz",
            u=self.u,
            v=self.v,
            w=self.w,
            eta=self.eta,
            nodes=nodes,
        )
        logging.info("Parameters saved in: %s", output_parameters + ".npz")
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')

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

        # Create a Path object for the output directory
        out_folder_path = Path(self.out_folder)

        # Create output dir if it does not exist
        out_folder_path.mkdir(parents=True, exist_ok=True)

        # Check if the output file name is provided
        if outfile is None:
            # If not provided, generate a default file name using the seed and average degree
            outfile = "syn" + str(self.seed) + "_k" + str(int(self.k)) + ".dat"

        # Get the list of edges from the graph along with their data
        edges = list(G.edges(data=True))

        try:
            # Try to extract the weight of each edge
            data = [[u, v, d["weight"]] for u, v, d in edges]
        except KeyError:
            # If the weight is not available, assign a default weight of 1
            data = [[u, v, 1] for u, v, d in edges]

        # Create a DataFrame from the edge data
        df = pd.DataFrame(data, columns=["source", "target", "w"], index=None)

        # Save the DataFrame to a CSV file
        df.to_csv(self.out_folder + outfile, index=False, sep=" ")

        logging.info("Adjacency matrix saved in: %s", self.out_folder + outfile)


def affinity_matrix(
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
    structure : str, optional
        Structure of the network (default is 'assortative').
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
    if structure == "assortative":
        # Assortative structure: higher probability within groups
        p = p1 * a * np.ones((K, K))  # secondary probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary probabilities

    elif structure == "disassortative":
        # Disassortative structure: higher probability between groups
        p = p1 * np.ones((K, K))  # primary probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary probabilities

    elif structure == "core-periphery":
        # Core-periphery structure: core nodes have higher probability
        p = p1 * np.ones((K, K))
        np.fill_diagonal(np.fliplr(p), a * p1)
        p[1, 1] = b * p1

    elif structure == "directed-biased":
        # Directed-biased structure: directional bias in probabilities
        p = a * p1 * np.ones((K, K))
        p[0, 1] = p1
        p[1, 0] = b * p1

    return p
