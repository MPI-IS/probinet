"""
Class definition of the reciprocity generative model with the member functions required.
It builds a directed, possibly weighted, network.
"""

from abc import ABCMeta
import logging
import math
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import RandomState
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse import tril, triu

from . import tools as tl
from ..model.constants import EPS_
from ..output.evaluate import lambda_full
from ..output.plot import plot_A
from .stats import print_graph_stat, reciprocal_edges
from .tools import (
    check_symmetric, Exp_ija_matrix, log_and_raise_error, normalize_nonzero_membership,
    output_adjacency, transpose_tensor)

# TODO: add type hints into a separate script

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

DEFAULT_OUT_FOLDER = "data/input/synthetic/"

DEFAULT_SHOW_DETAILS = True
DEFAULT_SHOW_PLOTS = False
DEFAULT_OUTPUT_NET = True

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GM_reciprocity:  # this could be called CRep (synthetic.CRep)
    """
    A class to generate a directed, possibly weighted, network with reciprocity.
    """

    def __init__(
        self,
        N: int,
        K: int,
        eta: float = 0.5,
        k: float = 3,
        ExpM: Optional[float] = None,
        over: float = 0.0,
        corr: float = 0.0,
        seed: int = 0,
        alpha: float = 0.1,
        ag: float = 0.1,
        beta: float = 0.1,
        Normalization: int = 0,
        structure: str = "assortative",
        end_file: str = "",
        out_folder: str = "../data/output/real_data/cv/",
        output_parameters: bool = False,
        output_adj: bool = False,
        outfile_adj: str = "None",
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
    ) -> Tuple[nx.MultiDiGraph, np.ndarray]:
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
        prng = np.random.RandomState(self.seed)  # pylint: disable=no-member

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
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1.0 / self.beta, size=(overlapping, self.K)
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.gamma(self.ag, 1.0 / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = tl.Exp_ija_matrix(
            self.u, self.v, self.w
        )  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)

        # Compute the constant to enforce sparsity in the network
        c = (self.ExpM * (1.0 - self.eta)) / M0.sum()

        # Compute the expected number of edges between each pair of nodes considering reciprocity
        MM = (M0 + self.eta * tl.transpose_ij2(M0)) / (
            1.0 - self.eta * self.eta
        )  # whose elements are m_{ij}
        Mt = tl.transpose_ij2(MM)
        MM0 = M0.copy()  # to be not influenced by c_lambda

        # Adjust the affinity matrix w and the expected number of edges M0 by the constant c
        if parameters is None:
            self.w *= c  # only w is impact by that, u and v have a constraint,
            # their sum over k should sum to 1
        M0 *= c
        M0t = tl.transpose_ij2(M0)  # whose elements are lambda0_{ji}

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
            "Sum of weights in the upper triangular matrix: %s",
            np.round(triu(A, k=1).sum(), 2),
        )
        logging.info(
            "Sum of weights in the lower triangular matrix: %s",
            np.round(tril(A, k=-1).sum(), 2),
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
        logging.info("Expected reciprocity: %s", np.round(Exp_r, 3))
        logging.info("Reciprocity (networkX) = %s", np.round(nx.reciprocity(G), 3))
        logging.info(
            "Reciprocity (considering the weights of the edges) = %s", np.round(rw, 3)
        )

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
        prng = np.random.RandomState(self.seed)  # pylint: disable=no-member

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
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1.0 / self.beta, size=(overlapping, self.K)
                    )
                    self.v[ind_over] = self.corr * self.u[ind_over] + (
                        1.0 - self.corr
                    ) * prng.gamma(self.ag, 1.0 / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = tl.Exp_ija_matrix(
            self.u, self.v, self.w
        )  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)
        M0t = tl.transpose_ij2(M0)  # whose elements are lambda0_{ji}

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
        logging.info("Expected reciprocity: %s", np.round(rw, 3))
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
        prng = np.random.RandomState(self.seed)  # pylint: disable=no-member

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
    a: float = 0.1,
    b: float = 0.3,
) -> np.ndarray:
    """
    Return the KxK affinity matrix w with probabilities between and within groups.

    Parameters
    ----------
    structure : str
                Structure of the network, e.g. assortative, disassortative.
    N : int
        Number of nodes.
    K : int
        Number of communities.
    a : float
        Parameter for secondary probabilities.
    b : float
        Parameter for third probabilities.

    Returns
    -------
    p : np.ndarray
        Array with probabilities between and within groups. Element (k,q) gives the density
        of edges going from the nodes of group k to nodes of group q.
    """

    b *= a
    p1 = K / N
    if structure == "assortative":
        p = p1 * a * np.ones((K, K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == "disassortative":
        p = p1 * np.ones((K, K))  # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    else:
        message = (
            "The structure of the affinity matrix w can be either assortative or "
            "disassortative!"
        )
        log_and_raise_error(ValueError, message)

    return p


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
        out_folder: str = DEFAULT_OUT_FOLDER,
        output_net: bool = DEFAULT_OUTPUT_NET,
        show_details: bool = DEFAULT_SHOW_DETAILS,
        show_plots: bool = DEFAULT_SHOW_PLOTS,
        **kwargs,  # this is needed later on
    ):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities

        # Set seed random number generator
        self.seed = seed
        self.eta = eta
        self.prng = np.random.RandomState(self.seed)

        self.out_folder = out_folder
        self.output_net = output_net

        self.show_details = show_details
        self.show_plots = show_plots


class StandardMMSBM(BaseSyntheticNetwork):
    """
    Create a synthetic, directed, and weighted network (possibly multilayer)
    by a standard mixed-membership stochastic block-model
    - It models marginals (iid assumption) with Poisson distributions
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        parameters = kwargs.get("parameters")

        self.init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        if self.output_net:
            self._output_parameters()
            output_adjacency(self.layer_graphs, self.out_folder, self.label)

        if self.show_details:
            print_graph_stat(self.G)
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

        if "label" in kwargs:
            label = kwargs["label"]
        else:
            try:
                message = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_eta_seed"
                logging.warning(message)
                label = "_".join(
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
                label = "_".join(
                    [
                        str(),
                        str(self.N),
                        str(self.L),
                        str(self.K),
                        str(self.avg_degree),
                        str(self.seed),
                    ]
                )
        self.label = label

        # SETUP overlapping communities

        if "perc_overlapping" in kwargs:
            perc_overlapping = kwargs["perc_overlapping"]
            if (perc_overlapping < 0) or (
                perc_overlapping > 1
            ):  # fraction of nodes with mixed membership
                log_and_raise_error(
                    ValueError,
                    "The percentage of overlapping nodes has to be in  " "[0, 1]!",
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
                    ValueError, "The parameter alpha should be a list of " "length K."
                )
            if not all(alpha):
                log_and_raise_error(
                    ValueError,
                    "Each entry of the Dirichlet parameter has to be " "positive!",
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

    def build_Y(self, parameters=None) -> None:
        """
        Generate network layers G using the latent variables,
        with the generative model A_ij ~ P(A_ij|u,v,w)

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
            if self.u.shape != (self.N, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter u has to be (N, K)."
                )

            if self.v.shape != (self.N, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter v has to be (N, K)."
                )

            if self.w.shape != (self.L, self.K, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter w has to be (L, K, K)."
                )

        # Generate Y

        self.M = Exp_ija_matrix(self.u, self.v, self.w)
        for layer in range(self.L):
            np.fill_diagonal(self.M[layer], 0)
        # sparsity parameter for Y
        if self.is_sparse:
            c = self.ExpEdges / self.M.sum()
            self.M *= c
            if parameters is None:
                self.w *= c

        Y = self.prng.poisson(self.M)

        # Create networkx DiGraph objects for each layer for easier manipulation

        nodes_to_remove = []
        self.G = []
        self.layer_graphs = []
        for layer in range(self.L):
            self.G.append(nx.from_numpy_array(Y[layer], create_using=nx.DiGraph()))
            Gc = max(nx.weakly_connected_components(self.G[layer]), key=len)
            nodes_to_remove.append(set(self.G[layer].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for layer in range(self.L):
            self.G[layer].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[layer].nodes())

            self.layer_graphs.append(
                nx.to_scipy_sparse_array(self.G[layer], nodelist=self.nodes)
            )

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
        ind_over = self.prng.randint(len(u), size=overlapping)

        u[ind_over] = self.prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
        v[ind_over] = self.correlation_u_v * u[ind_over] + (
            1.0 - self.correlation_u_v
        ) * self.prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
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

    def _output_parameters(self) -> None:
        """
        Output results in a compressed file.
        """

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        output_parameters = self.out_folder + "gt_" + self.label
        try:
            np.savez_compressed(
                output_parameters + ".npz",
                u=self.u,
                v=self.v,
                w=self.w,
                eta=self.eta,
                nodes=self.nodes,
            )
        except AttributeError:
            np.savez_compressed(
                output_parameters + ".npz",
                u=self.u,
                v=self.v,
                w=self.w,
                nodes=self.nodes,
            )
        logging.info("Parameters saved in: %s", output_parameters + ".npz")
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')

    # pylint: disable=W0631
    def _plot_M(self, cmap: str = "PuBuGn") -> None:
        """
        Plot the marginal means produced by the generative algorithm.

        Parameters
        ----------
        M : ndarray
            Mean lambda for all entries.
        """

        for layer in range(self.L):
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(self.M[layer], cmap=plt.get_cmap(cmap))
            ax.set_title(f"Marginal means matrix layer {layer}", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    # pylint: enable=W0631


class ReciprocityMMSBM_joints(StandardMMSBM):
    """
    Proposed benchmark.
    Create a synthetic, directed, and binary network (possibly multilayer)
    by a mixed-membership stochastic block-model with a reciprocity structure
    - It models pairwise joint distributions with Bivariate Bernoulli distributions
    """

    def __init__(self, **kwargs):  # pylint: disable=super-init-not-called
        # TODO: incorporate the __init__ where it should

        if "eta" in kwargs:
            eta = kwargs["eta"]
            if eta <= 0:  # pair interaction coefficient
                message = (
                    "The pair interaction coefficient eta has to be greater than 0!"
                )
                log_and_raise_error(ValueError, message)
        else:
            message = f"eta parameter was not set. Defaulting to eta={DEFAULT_ETA}"
            logging.warning(message)
            eta = DEFAULT_ETA
        self.eta = eta

        parameters = kwargs.get("parameters")

        super().init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        if self.output_net:
            super()._output_parameters()
            output_adjacency(self.layer_graphs, self.out_folder, self.label)

        if self.show_details:
            print_graph_stat(self.G)
        if self.show_plots:
            plot_A(self.layer_graphs)
            if self.M0 is not None:
                self._plot_M()

    def build_Y(
        self, parameters: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Generate network layers G using the latent variables,
        with the generative model (A_ij,A_ji) ~ P(A_ij, A_ji|u,v,w,eta)

        Parameters
        ----------
        parameters: Tuple[np.ndarray, np.ndarray, np.ndarray], optional
                    Latent variables u, v, and w.
        """

        # Latent variables

        if parameters is None:
            # generate latent variables
            self.u, self.v, self.w = self._generate_lv()
        else:
            # set latent variables
            self.u, self.v, self.w = parameters
            if self.u.shape != (self.N, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter u has to be (N, K)."
                )

            if self.v.shape != (self.N, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter v has to be (N, K)."
                )

            if self.w.shape != (self.L, self.K, self.K):
                log_and_raise_error(
                    ValueError, "The shape of the parameter w has to be (L, K, K)."
                )

        # Generate Y

        self.G = [nx.DiGraph() for _ in range(self.L)]
        self.layer_graphs = []

        nodes_to_remove = []
        for layer in range(self.L):
            for i in range(self.N):
                self.G[layer].add_node(i)

        # whose elements are lambda0_{ij}
        self.M0 = lambda_full(self.u, self.v, self.w)
        for layer in range(self.L):
            np.fill_diagonal(self.M0[layer], 0)
            if self.is_sparse:
                # constant to enforce sparsity
                c = brentq(
                    self._eq_c,
                    0.00001,
                    100.0,
                    args=(self.ExpEdges, self.M0[layer], self.eta),
                )
                self.M0[layer] *= c
                if parameters is None:
                    self.w[layer] *= c
        # compute the normalization constant
        self.Z = self._calculate_Z(self.M0, self.eta)

        for layer in range(self.L):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # The probabilities look like [p00, p01, p10, p11]
                    probabilities = (
                        np.array(
                            [
                                1.0,
                                self.M0[layer, j, i],
                                self.M0[layer, i, j],
                                self.M0[layer, i, j] * self.M0[layer, j, i] * self.eta,
                            ]
                        )
                        / self.Z[layer, i, j]
                    )
                    cumulative = [
                        1.0 / self.Z[layer, i, j],
                        np.sum(probabilities[:2]),
                        np.sum(probabilities[:3]),
                        1.0,
                    ]

                    r = self.prng.rand(1)[0]
                    if r <= probabilities[0]:
                        A_ij, A_ji = 0, 0
                    elif probabilities[0] < r <= cumulative[1]:
                        A_ij, A_ji = 0, 1
                    elif cumulative[1] < r <= cumulative[2]:
                        A_ij, A_ji = 1, 0
                    elif r > cumulative[2]:
                        A_ij, A_ji = 1, 1
                    if A_ij > 0:
                        self.G[layer].add_edge(i, j, weight=1)  # binary
                    if A_ji > 0:
                        self.G[layer].add_edge(j, i, weight=1)  # binary

            assert len(list(self.G[layer].nodes())) == self.N

            # keep largest connected component
            Gc = max(nx.weakly_connected_components(self.G[layer]), key=len)
            nodes_to_remove.append(set(self.G[layer].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for layer in range(self.L):
            self.G[layer].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[layer].nodes())

            self.layer_graphs.append(
                nx.to_scipy_sparse_array(self.G[layer], nodelist=self.nodes)
            )

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)

    def _calculate_Z(self, lambda_aij: np.ndarray, eta: float) -> np.ndarray:
        """
        Compute the normalization constant of the Bivariate Bernoulli distribution.

        Parameters
        ----------
        lambda_aij : ndarray
                     Tensor with the mean lambda for all entries.
        eta : float
              Reciprocity coefficient.

        Returns
        -------
        Z : ndarray
            Normalization constant Z of the Bivariate Bernoulli distribution.
        """

        Z = (
            lambda_aij
            + transpose_tensor(lambda_aij)
            + eta * np.einsum("aij,aji->aij", lambda_aij, lambda_aij)
            + 1
        )
        check_symmetric(Z)

        return Z

    def _eq_c(self, c: float, ExpM: int, M: np.ndarray, eta: float) -> float:
        """
        Compute the function to set to zero to find the value of the sparsity parameter c.

        Parameters
        ----------
        c : float
            Sparsity parameter.
        ExpM : int
               In-coming membership matrix.
        M : ndarray
            Mean lambda for all entries.
        eta : float
              Reciprocity coefficient.

        Returns
        -------
        Value of the function to set to zero to find the value of the sparsity parameter c.
        """

        LeftHandSide = (c * M + c * c * eta * M * M.T) / (
            c * M + c * M.T + c * c * eta * M * M.T + 1.0
        )

        return np.sum(LeftHandSide) - ExpM

    # pylint: disable= W0631
    def _plot_M(self, cmap: str = "PuBuGn") -> None:
        """
        Plot the marginal means produced by the generative algorithm.

        Parameters
        ----------
        cmap : Matplotlib object
               Colormap used for the plot.
        """

        M = (self.M0 + self.eta * self.M0 * transpose_tensor(self.M0)) / self.Z
        for layer in range(self.L):
            np.fill_diagonal(M[layer], 0.0)
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(M[layer], cmap=plt.get_cmap(cmap))
            ax.set_title(f"Marginal means matrix layer {layer}", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    # pylint: enable=W0631


class CRepDyn:

    def __init__(
        self,
        N,
        K,
        T=1,
        eta=0.0,
        L=1,
        avg_degree=5.0,
        ExpM=None,
        prng=0,
        verbose=0,
        beta=0.2,
        ag=0.1,
        bg=0.1,
        eta_dir=0.5,
        L1=True,
        corr=1.0,
        over=0.0,
        label=None,
        end_file=".dat",
        undirected=False,
        folder="",
        structure="assortative",
        output_parameters=False,
        output_adj=False,
        outfile_adj=None,
    ):
        self.N = N
        self.K = K
        self.T = T
        self.L = L
        self.avg_degree = avg_degree
        self.prng = prng
        self.end_file = end_file
        self.undirected = undirected
        self.folder = folder
        self.output_parameters = output_parameters
        self.output_adj = output_adj
        self.outfile_adj = outfile_adj

        if label is not None:
            self.label = label
        else:
            self.label = ("_").join(
                [str(N), str(K), str(avg_degree), str(T), str(eta), str(beta)]
            )
        self.structure = structure
        print("=" * 30)
        print("self.structure:", self.structure)

        if ExpM is None:
            self.ExpM = self.avg_degree * self.N * 0.5
        else:
            self.ExpM = float(ExpM)

        # Set verbosity flag
        if verbose not in {0, 1, 2}:
            raise ValueError(
                "The verbosity parameter can only assume values in {0,1,2}!"
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
        Exp_ija = np.einsum("ik,kq->iq", u, w)
        Exp_ija = np.einsum("iq,jq->ij", Exp_ija, v)
        return Exp_ija

    def CRepDyn_network(self, parameters=None):
        """
        Generate a directed, possibly weighted network by using CRep Dyn
        Steps:
            1. Generate a network A[0]
            2. Extract A[t] entries (network edges) using transition probabilities
        INPUT
        ----------
        parameters : object
                     Latent variables eta, beta, u, v and w.
        OUTPUT
        ----------
        G : Digraph
            DiGraph NetworkX object. Self-loops allowed.
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
            print(
                f"Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component"
            )

        if self.verbose > 0:
            for t in range(len(G)):
                print("-" * 30)
                print("t=", t)
                ave_w_deg = np.round(
                    2 * G[t].number_of_edges() / float(G[t].number_of_nodes()), 3
                )
                print(
                    f"Number of nodes: {G[t].number_of_nodes()} \n"
                    f"Number of edges: {G[t].number_of_edges()}"
                )
                print(f"Average degree (2E/N): {ave_w_deg}")
                print(f"Reciprocity at t: {nx.reciprocity(G[t])}")
                print("-" * 30)

            self.check_reciprocity_tm1(A, A_sum)

        if self.output_parameters:
            self._output_results(nodes)

        if self.verbose == 2:
            self._plot_A(A)
            if M is not None:
                self._plot_M(M)

        return G

    def _generate_lv(self, prng=42):
        """
        Generate z, u, v, w latent variables.
        INPUT
        ----------
        prng : int
               Seed for the random number generator.
        OUTPUT
        ----------
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

    def _build_multilayer_edgelist(self, nodes, A_tot, A, nodes_to_keep=None):
        A_coo = A_tot.tocoo()
        data_dict = {"source": A_coo.row, "target": A_coo.col}
        for t in range(len(A)):
            data_dict["weight_t" + str(t)] = np.squeeze(
                np.asarray(A[t][A_tot.nonzero()])
            )

        df_res = pd.DataFrame(data_dict)
        print(len(df_res))
        if nodes_to_keep is not None:
            df_res = df_res[
                df_res.source.isin(nodes_to_keep) & df_res.target.isin(nodes_to_keep)
            ]

        nodes = list(set(df_res.source).union(set(df_res.target)))
        id2node = {}
        for i, n in enumerate(nodes):
            id2node[i] = n

        df_res["source"] = df_res.source.map(id2node)
        df_res["target"] = df_res.target.map(id2node)

        return df_res

    def _output_results(self, nodes):
        """
        Output results in a compressed file.
        INPUT
        ----------
        nodes : list
                List of nodes IDs.
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
        if self.verbose:
            print()
            print(f"Parameters saved in: {output_parameters}.npz")
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _output_adjacency(self, nodes, A_tot, A, nodes_to_keep=None, outfile=None):
        """
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

        df = self._build_multilayer_edgelist(
            nodes, A_tot, A, nodes_to_keep=nodes_to_keep
        )
        df.to_csv(self.folder + outfile, index=False, sep=" ")
        if self.verbose:
            print(f"Adjacency matrix saved in: {self.folder + outfile}")

    def _plot_A(self, A, cmap="PuBuGn"):
        """
        Plot the adjacency matrix produced by the generative algorithm.
        INPUT
        ----------
        A : Scipy array
            Sparse version of the NxN adjacency matrix associated to the graph.
        cmap : Matplotlib object
               Colormap used for the plot.
        """
        for i in range(len(A)):
            Ad = A[i].todense()
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(Ad, cmap=plt.get_cmap(cmap))
            ax.set_title("Adjacency matrix", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    def _plot_M(self, M, cmap="PuBuGn"):
        """
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

        _, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(M, cmap=plt.get_cmap(cmap))
        ax.set_title("MT means matrix", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def check_reciprocity_tm1(self, A):
        for t in range(1, len(A)):
            ref_subs = A[t].nonzero()
            M_t_T = A[t].transpose()[ref_subs]
            M_tm1_T = A[t - 1].transpose()[ref_subs]
            nnz = float(A[t].count_nonzero())
            print(
                nnz,
                M_t_T.nonzero()[0].shape[0] / nnz,
                M_tm1_T.nonzero()[0].shape[0] / nnz,
            )


def membership_vectors(
    prng: RandomState = RandomState(10),
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
    INPUT
    ----------
    prng: Numpy Random object
          Random number generator container.
    L1 : bool
         Flag for parameter generation method. True for Dirichlet, False for Gamma.
    eta : float
          Parameter for Dirichlet.
    alpha : float
        Parameter (alpha) for Gamma.
    beta : float
        Parameter (beta) for Gamma.
    N : int
        Number of nodes.
    K : int
        Number of communities.
    corr : float
           Correlation between u and v synthetically generated.
    over : float
           Fraction of nodes with mixed membership.
    OUTPUT
    -------
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
            )  # TODO: Ask Hadiseh
            # why eta is not defined
            v[ind_over] = corr * u[ind_over] + (1.0 - corr) * prng.dirichlet(
                eta * np.ones(K), overlapping  # pylint: disable=undefined-variable
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
