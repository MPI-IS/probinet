"""
Class definition of the reciprocity generative model with the member functions required.
It builds a directed, possibly weighted, network.
"""
from abc import ABCMeta
import math
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import warnings

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse import tril, triu

from . import tools as tl
from ..model.jointcrep import transpose_tensor
from ..output.evaluate import _lambda0_full
from ..output.plot import plot_A
from .stats import print_graph_stat, reciprocal_edges
from .tools import check_symmetric, Exp_ija_matrix, normalize_nonzero_membership, output_adjacency

# TODO: add type hints into a separte script

DEFAULT_N = 1000
DEFAULT_L = 1
DEFAULT_K = 2
DEFAULT_ETA = 50
DEFAULT_ALPHA_HL = 6
DEFAULT_AVG_DEGREE = 15
DEFAULT_STRUCTURE = "assortative"

DEFAULT_PERC_OVERLAPPING = 0.2
DEFAULT_CORRELATION_U_V = 0.
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

    def __init__(self,
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
                 structure: str = 'assortative',
                 end_file: str = '',
                 out_folder: str = '../data/output/real_data/cv/',
                 output_parameters: bool = False,
                 output_adj: bool = False,
                 outfile_adj: str = 'None',
                 verbose: bool = False):
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
        self.verbose = verbose  # flag to print details
        if (eta < 0) or (eta >= 1):  # reciprocity coefficient
            raise ValueError(
                'The reciprocity coefficient eta has to be in [0, 1)!')
        self.eta = eta
        if ExpM is None:  # expected number of edges
            self.ExpM = int(self.N * self.k / 2.)
        else:
            self.ExpM = int(ExpM)
            self.k = 2 * self.ExpM / float(self.N)
        if (over < 0) or (over > 1):  # fraction of nodes with mixed membership
            raise ValueError('The over parameter has to be in [0, 1]!')
        self.over = over
        if (corr < 0) or (
                corr
                > 1):  # correlation between u and v synthetically generated
            raise ValueError(
                'The correlation parameter corr has to be in [0, 1]!')
        self.corr = corr
        if Normalization not in {
            0, 1
        }:  # indicator for choosing how to generate the latent variables
            raise ValueError(
                'The Normalization parameter can be either 0 or 1! It is used as an indicator for '
                'generating the membership matrices u and v from a Dirichlet or a Gamma '
                'distribution, respectively. It is used when there is overlapping.')
        self.Normalization = Normalization
        if structure not in {'assortative', 'disassortative'
                             }:  # structure of the affinity matrix W
            raise ValueError(
                'The structure of the affinity matrix w can be either assortative or '
                'disassortative!')
        self.structure = structure

    def reciprocity_planted_network(
            self,
            parameters: Optional[Tuple[np.ndarray,
                                       np.ndarray,
                                       np.ndarray,
                                       float]] = None
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
        G: MultiDigraph
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
                    self.u[i:, self.K - 1] = 1.
                    self.v[i:, self.K - 1] = 1.
                else:
                    # Assign the current community to the nodes in the current range
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.
                        self.v[j, q] = 1.

            # Generate the affinity matrix w
            self.w = affinity_matrix(structure=self.structure,
                                     N=self.N,
                                     K=self.K,
                                     a=0.1,
                                     b=0.3)

            # Check if there is overlapping in the communities
            if self.over != 0.:
                # Calculate the number of nodes belonging to more communities
                overlapping = int(self.N * self.over)
                # Randomly select 'overlapping' number of nodes
                ind_over = np.random.randint(len(self.u), size=overlapping)

                # Check the normalization method
                if self.Normalization == 0:
                    # If Normalization is 0, generate u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                        prng.dirichlet(self.alpha * np.ones(self.K), overlapping)

                    # If correlation is 1, ensure u and v are close
                    if self.corr == 1.:
                        assert np.allclose(self.u, self.v)

                    # If correlation is greater than 0, normalize v
                    if self.corr > 0:
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                        prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = tl.Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
        np.fill_diagonal(M0, 0)

        # Compute the constant to enforce sparsity in the network
        c = (self.ExpM * (1. - self.eta)) / M0.sum()

        # Compute the expected number of edges between each pair of nodes considering reciprocity
        MM = (M0 + self.eta * tl.transpose_ij2(M0)) / \
             (1. - self.eta * self.eta)  # whose elements are m_{ij}
        Mt = tl.transpose_ij2(MM)
        MM0 = M0.copy()  # to be not influenced by c_lambda

        # Adjust the affinity matrix w and the expected number of edges M0 by the constant c
        if parameters is None:
            self.w *= c  # only w is impact by that, u and v have a constraint,
            # their sum over k should sum to 1
        M0 *= c
        M0t = tl.transpose_ij2(M0)  # whose elements are lambda0_{ji}

        # Compute the expected number of edges between each pair of nodes considering reciprocity
        M = (M0 + self.eta * M0t) / (1. - self.eta * self.eta)  # whose elements are m_{ij}
        np.fill_diagonal(M, 0)

        # Compute the expected reciprocity in the network
        rw = self.eta + ((MM0 * Mt + self.eta * Mt ** 2).sum() / MM.sum())  # expected reciprocity

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
                    A_ij = prng.poisson(M[i, j], 1)[0]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ij > 0:
                        G.add_edge(i, j, weight=A_ij)
                    # Compute the expected number of edges from node j to node i considering
                    # reciprocity
                    lambda_ji = M0[j, i] + self.eta * A_ij
                    # Draw the number of edges from node j to node i from a Poisson distribution
                    A_ji = prng.poisson(
                        lambda_ji, 1
                    )[0]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                else:
                    # Draw the number of edges from node j to node i from a Poisson distribution
                    A_ji = prng.poisson(M[j, i], 1)[0]  # draw A_ij from P(A_ij) = Poisson(m_ij)
                    if A_ji > 0:
                        G.add_edge(j, i, weight=A_ji)
                    # Compute the expected number of edges from node i to node j considering
                    # reciprocity
                    lambda_ij = M0[i, j] + self.eta * A_ji
                    # Draw the number of edges from node i to node j from a Poisson distribution
                    A_ij = prng.poisson(
                        lambda_ij, 1
                    )[0]  # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
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
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight')

        # Compute the average degree and the average weighted degree in the network
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Compute the proportion of bi-directional edges over the unordered pairs of nodes
        reciprocity_c = np.round(reciprocal_edges(G), 3)

        # Print the details of the network if verbose is True
        if self.verbose:
            print(
                f'Number of links in the upper triangular matrix: {triu(A, k=1).nnz}\n'
                f'Number of links in the lower triangular matrix: {tril(A, k=-1).nnz}'
            )
            print(
                f'Sum of weights in the upper triangular matrix: '
                f'{np.round(triu(A, k=1).sum(), 2)}\n'
                f'Sum of weights in the lower triangular matrix: '
                f'{np.round(tril(A, k=-1).sum(), 2)}\n'
                f'Number of possible unordered pairs: {counter}')
            print(
                f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected '
                f'component'
            )
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(f'Expected reciprocity: {np.round(rw, 3)}')
            print(
                f'Reciprocity (intended as the proportion of bi-directional edges over the  '
                f'unordered pairs): '
                f'{reciprocity_c}\n')

        # Output the parameters of the network if output_parameters is True
        if self.output_parameters:
            self.output_results(nodes)

        # Output the adjacency matrix of the network if output_adj is True
        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G, A

    def planted_network_cond_independent(self,
                                         parameters: Optional[Tuple[np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray]] = None) -> Tuple[nx.MultiDiGraph,
                                                                                                  np.ndarray]:
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
                    self.u[i:, self.K - 1] = 1.
                    self.v[i:, self.K - 1] = 1.
                else:
                    # Assign the current community to the nodes in the current range
                    for j in range(q * size, q * size + size):
                        self.u[j, q] = 1.
                        self.v[j, q] = 1.

            # Generate the affinity matrix w
            self.w = affinity_matrix(structure=self.structure,
                                     N=self.N,
                                     K=self.K,
                                     a=0.1,
                                     b=0.3)

            # Check if there is overlapping in the communities
            if self.over != 0.:
                # Calculate the number of nodes belonging to more communities
                overlapping = int(self.N * self.over)
                # Randomly select 'overlapping' number of nodes
                ind_over = np.random.randint(len(self.u), size=overlapping)

                # Check the normalization method
                if self.Normalization == 0:
                    # If Normalization is 0, generate u and v from a Dirichlet distribution
                    self.u[ind_over] = prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                        prng.dirichlet(self.alpha * np.ones(self.K), overlapping)

                    # If correlation is 1, ensure u and v are close
                    if self.corr == 1.:
                        assert np.allclose(self.u, self.v)

                    # If correlation is greater than 0, normalize v
                    if self.corr > 0:
                        self.v = tl.normalize_nonzero_membership(self.v)
                elif self.Normalization == 1:
                    # If Normalization is 1, generate u and v from a Gamma distribution
                    self.u[ind_over] = prng.gamma(
                        self.ag, 1. / self.beta, size=(overlapping, self.K))
                    self.v[ind_over] = self.corr * self.u[ind_over] + (1. - self.corr) * \
                        prng.gamma(self.ag, 1. / self.beta, size=(overlapping, self.K))

                    # Normalize u and v
                    self.u = tl.normalize_nonzero_membership(self.u)
                    self.v = tl.normalize_nonzero_membership(self.v)

        # Compute the expected number of edges between each pair of nodes
        M0 = tl.Exp_ija_matrix(self.u, self.v, self.w)  # whose elements are lambda0_{ij}
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
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight')

        # Calculate the average degree and the average weighted degree in the graph
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Calculate the proportion of bi-directional edges over the unordered pairs of nodes
        reciprocity_c = np.round(reciprocal_edges(G), 3)

        # Print the details of the network if verbose is True
        if self.verbose:
            print(
                f'Number of links in the upper triangular matrix: {triu(A, k=1).nnz}\n'
                f'Number of links in the lower triangular matrix: {tril(A, k=-1).nnz}'
            )
            print(
                f'Sum of weights in the upper triangular matrix: '
                f'{np.round(triu(A, k=1).sum(), 2)}\n'
                f'Sum of weights in the lower triangular matrix: '
                f'{np.round(tril(A, k=-1).sum(), 2)}')
            print(
                f'Removed {len(nodes_to_remove)} nodes, because not part of the largest '
                f'connected component'
            )
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(f'Expected reciprocity: {np.round(rw, 3)}')
            print(
                f'Reciprocity (intended as the proportion of bi-directional edges over the '
                f'unordered pairs): '
                f'{reciprocity_c}\n')

        # Output the parameters of the network if output_parameters is True
        if self.output_parameters:
            self.output_results(nodes)

        # Output the adjacency matrix of the network if output_adj is True
        if self.output_adj:
            self.output_adjacency(G, outfile=self.outfile_adj)

        return G, A

    def planted_network_reciprocity_only(
            self, p: Optional[float] = None) -> Tuple[nx.MultiDiGraph, np.ndarray]:
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
            p = (1. - self.eta) * self.k * 0.5 / (self.N - 1.)

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
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight')

        # Calculate the average degree and the average weighted degree in the graph
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
        ave_w_deg = np.round(2 * totM / float(G.number_of_nodes()), 3)

        # Calculate the proportion of bi-directional edges over the unordered pairs of nodes
        reciprocity_c = np.round(reciprocal_edges(G), 3)

        # Print the details of the graph if verbose is True
        if self.verbose:
            print(
                f'Number of links in the upper triangular matrix: {triu(A, k=1).nnz}\n'
                f'Number of links in the lower triangular matrix: {tril(A, k=-1).nnz}'
            )
            print(
                f'Sum of weights in the upper triangular matrix: '
                f'{np.round(triu(A, k=1).sum(), 2)}\n'
                f'Sum of weights in the lower triangular matrix: '
                f'{np.round(tril(A, k=-1).sum(), 2)}')
            print(
                f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected '
                f'component'
            )
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {Sparsity_cof}')
            print(f'Average weighted degree (2M/N): {ave_w_deg}')
            print(
                f'Reciprocity (intended as the proportion of bi-directional edges over the '
                f'unordered pairs): '
                f'{reciprocity_c}\n')

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

        output_parameters = self.out_folder + 'theta_gt' + str(
            self.seed) + self.end_file
        np.savez_compressed(output_parameters + '.npz',
                            u=self.u,
                            v=self.v,
                            w=self.w,
                            eta=self.eta,
                            nodes=nodes)
        if self.verbose:
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def output_adjacency(self, G: nx.MultiDiGraph, outfile: Optional[str] = None) -> None:
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
            outfile = 'syn' + str(self.seed) + '_k' + str(int(self.k)) + '.dat'

        # Get the list of edges from the graph along with their data
        edges = list(G.edges(data=True))

        try:
            # Try to extract the weight of each edge
            data = [[u, v, d['weight']] for u, v, d in edges]
        except KeyError:
            # If the weight is not available, assign a default weight of 1
            data = [[u, v, 1] for u, v, d in edges]

        # Create a DataFrame from the edge data
        df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)

        # Save the DataFrame to a CSV file
        df.to_csv(self.out_folder + outfile, index=False, sep=' ')

        # If verbose mode is enabled, print the location of the saved file
        if self.verbose:
            print(f'Adjacency matrix saved in: {self.out_folder + outfile}')


def affinity_matrix(structure: str = 'assortative',
                    N: int = 100,
                    K: int = 2,
                    a: float = 0.1,
                    b: float = 0.3) -> np.ndarray:
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
    if structure == 'assortative':
        p = p1 * a * np.ones((K, K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == 'disassortative':
        p = p1 * np.ones((K, K))  # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    else:
        raise ValueError(
            'The structure of the affinity matrix w can be either assortative or disassortative!')

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
            out_folder: str = DEFAULT_OUT_FOLDER,
            output_net: bool = DEFAULT_OUTPUT_NET,
            show_details: bool = DEFAULT_SHOW_DETAILS,
            show_plots: bool = DEFAULT_SHOW_PLOTS,
            **kwargs  # this is needed later on
    ):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities

        # Set seed random number generator
        self.seed = seed
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
                err_msg = "The average degree has to be greater than 0.!"
                raise ValueError(err_msg)
        else:
            msg = f"avg_degree parameter was not set. Defaulting to avg_degree={DEFAULT_AVG_DEGREE}"
            warnings.warn(msg)
            avg_degree = DEFAULT_AVG_DEGREE
        self.avg_degree = avg_degree
        self.ExpEdges = int(self.avg_degree * self.N * 0.5)

        if "is_sparse" in kwargs:
            is_sparse = kwargs["is_sparse"]
        else:
            msg = f"is_sparse parameter was not set. Defaulting to is_sparse={DEFAULT_IS_SPARSE}"
            warnings.warn(msg)
            is_sparse = DEFAULT_IS_SPARSE
        self.is_sparse = is_sparse

        if "label" in kwargs:
            label = kwargs["label"]
        else:
            try:
                msg = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_eta_seed"
                warnings.warn(msg)
                label = '_'.join(
                    [
                        str(), str(
                            self.N), str(
                            self.L), str(
                            self.K), str(
                            self.avg_degree), str(
                            self.eta), str(
                            self.seed)])
            except AttributeError:
                msg = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree_seed"
                warnings.warn(msg)
                label = '_'.join([str(), str(self.N), str(self.L), str(
                    self.K), str(self.avg_degree), str(self.seed)])
        self.label = label

        # SETUP overlapping communities

        if "perc_overlapping" in kwargs:
            perc_overlapping = kwargs["perc_overlapping"]
            if (perc_overlapping < 0) or (perc_overlapping >
                                          1):  # fraction of nodes with mixed membership
                err_msg = "The percentage of overlapping nodes has to be in [0, 1]!"
                raise ValueError(err_msg)
        else:
            msg = (f"perc_overlapping parameter was not set. Defaulting to perc_overlapping"
                   f"={DEFAULT_PERC_OVERLAPPING}")
            warnings.warn(msg)
            perc_overlapping = DEFAULT_PERC_OVERLAPPING
        self.perc_overlapping = perc_overlapping

        if self.perc_overlapping:
            # correlation between u and v synthetically generated
            if "correlation_u_v" in kwargs:
                correlation_u_v = kwargs["correlation_u_v"]
                if (correlation_u_v < 0) or (correlation_u_v > 1):
                    err_msg = "The correlation between u and v has to be in [0, 1]!"
                    raise ValueError(err_msg)
            else:
                msg = (f"correlation_u_v parameter for overlapping communities was not set. "
                       f"Defaulting to corr={DEFAULT_CORRELATION_U_V}")
                warnings.warn(msg)
                correlation_u_v = DEFAULT_CORRELATION_U_V
            self.correlation_u_v = correlation_u_v

            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            else:
                msg = (f"alpha parameter of Dirichlet distribution was not set. "
                       f"Defaulting to alpha={[DEFAULT_ALPHA] * self.K}")
                warnings.warn(msg)
                alpha = [DEFAULT_ALPHA] * self.K
            if isinstance(alpha, float):
                if alpha <= 0:
                    err_msg = "Each entry of the Dirichlet parameter has to be positive!"
                    raise ValueError(err_msg)

                alpha = [alpha] * self.K
            elif len(alpha) != self.K:
                err_msg = "The parameter alpha should be a list of length K."
                raise ValueError(err_msg)
            if not all(alpha):
                err_msg = "Each entry of the Dirichlet parameter has to be positive!"
                raise ValueError(err_msg)
            self.alpha = alpha

        # SETUP informed structure

        if "structure" in kwargs:
            structure = kwargs["structure"]
        else:
            msg = (f"structure parameter was not set. Defaulting to "
                   f"structure={[DEFAULT_STRUCTURE] * self.L}")
            warnings.warn(msg)
            structure = [DEFAULT_STRUCTURE] * self.L
        if isinstance(structure, str):
            if structure not in ["assortative", "disassortative"]:
                err_msg = ("The available structures for the affinity tensor w are: "
                           "assortative, disassortative!")
                raise ValueError(err_msg)
            structure = [structure] * self.L
        elif len(structure) != self.L:
            err_msg = ("The parameter structure should be a list of length L. "
                       "Each entry defines the structure of the corresponding layer!")
            raise ValueError(err_msg)
        for e in structure:
            if e not in ["assortative", "disassortative"]:
                err_msg = ("The available structures for the affinity tensor w are: "
                           "assortative, disassortative!")
                raise ValueError(err_msg)
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
                raise ValueError('The shape of the parameter u has to be (N, K).')
            if self.v.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter v has to be (N, K).')
            if self.w.shape != (self.L, self.K, self.K):
                raise ValueError('The shape of the parameter w has to be (L, K, K).')

        # Generate Y

        self.M = Exp_ija_matrix(self.u, self.v, self.w)
        for l in range(self.L):
            np.fill_diagonal(self.M[l], 0)
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
        for l in range(self.L):
            self.G.append(nx.from_numpy_array(Y[l], create_using=nx.DiGraph()))
            Gc = max(nx.weakly_connected_components(self.G[l]), key=len)
            nodes_to_remove.append(set(self.G[l].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for l in range(self.L):
            self.G[l].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[l].nodes())

            self.layer_graphs.append(nx.to_scipy_sparse_array(self.G[l], nodelist=self.nodes))

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)

    def _apply_overlapping(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        v[ind_over] = self.correlation_u_v * u[ind_over] + (1.0 - self.correlation_u_v) * \
            self.prng.dirichlet(self.alpha * np.ones(self.K), overlapping)
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
        for l in range(self.L):
            w[l, :, :] = self._compute_affinity_matrix(self.structure[l])

        return u, v, w

    def _output_parameters(self) -> None:
        """
        Output results in a compressed file.
        """

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        output_parameters = self.out_folder + 'gt_' + self.label
        try:
            np.savez_compressed(
                output_parameters +
                '.npz',
                u=self.u,
                v=self.v,
                w=self.w,
                eta=self.eta,
                nodes=self.nodes)
        except AttributeError:
            np.savez_compressed(
                output_parameters +
                '.npz',
                u=self.u,
                v=self.v,
                w=self.w,
                nodes=self.nodes)
        print(f'Parameters saved in: {output_parameters}.npz')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')

    # pylint: disable=W0631
    def _plot_M(self, cmap: str = 'PuBuGn') -> None:
        """
        Plot the marginal means produced by the generative algorithm.

        Parameters
        ----------
        M : ndarray
            Mean lambda for all entries.
        """

        for l in range(self.L):
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(self.M[l], cmap=plt.get_cmap(cmap))
            ax.set_title(f'Marginal means matrix layer {l}', fontsize=15)
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
                raise ValueError('The pair interaction coefficient eta has to greater than 0.!')
        else:
            msg = f"eta parameter was not set. Defaulting to eta={DEFAULT_ETA}"
            warnings.warn(msg)
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

    def build_Y(self, parameters: Optional[Tuple[np.ndarray,
                                                 np.ndarray, np.ndarray]] = None) -> None:
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
                raise ValueError('The shape of the parameter u has to be (N, K).')
            if self.v.shape != (self.N, self.K):
                raise ValueError('The shape of the parameter v has to be (N, K).')
            if self.w.shape != (self.L, self.K, self.K):
                raise ValueError('The shape of the parameter w has to be (L, K, K).')

        # Generate Y

        self.G = [nx.DiGraph() for _ in range(self.L)]
        self.layer_graphs = []

        nodes_to_remove = []
        for l in range(self.L):
            for i in range(self.N):
                self.G[l].add_node(i)

        # whose elements are lambda0_{ij}
        self.M0 = _lambda0_full(self.u, self.v, self.w)
        for l in range(self.L):
            np.fill_diagonal(self.M0[l], 0)
            if self.is_sparse:
                # constant to enforce sparsity
                c = brentq(self._eq_c, 0.00001, 100., args=(self.ExpEdges, self.M0[l], self.eta))
                # print(f'Constant to enforce sparsity: {np.round(c, 3)}')
                self.M0[l] *= c
                if parameters is None:
                    self.w[l] *= c
        # compute the normalization constant
        self.Z = self._calculate_Z(self.M0, self.eta)

        for l in range(self.L):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # [p00, p01, p10, p11]
                    probabilities = np.array([1., self.M0[l, j, i], self.M0[l, i, j],
                                              self.M0[l, i, j] * self.M0[l, j, i] * self.eta]) / \
                        self.Z[l, i, j]
                    cumulative = [1. / self.Z[l, i, j],
                                  np.sum(probabilities[:2]), np.sum(probabilities[:3]), 1.]

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
                        self.G[l].add_edge(i, j, weight=1)  # binary
                    if A_ji > 0:
                        self.G[l].add_edge(j, i, weight=1)  # binary

            assert len(list(self.G[l].nodes())) == self.N

            # keep largest connected component
            Gc = max(nx.weakly_connected_components(self.G[l]), key=len)
            nodes_to_remove.append(set(self.G[l].nodes()).difference(Gc))

        n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for l in range(self.L):
            self.G[l].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[l].nodes())

            self.layer_graphs.append(nx.to_scipy_sparse_array(self.G[l], nodelist=self.nodes))

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

        Z = lambda_aij + transpose_tensor(lambda_aij) + eta * \
            np.einsum('aij,aji->aij', lambda_aij, lambda_aij) + 1
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

        LeftHandSide = (c * M + c * c * eta * M * M.T) / \
                       (c * M + c * M.T + c * c * eta * M * M.T + 1.)

        return np.sum(LeftHandSide) - ExpM

    # pylint: disable= W0631
    def _plot_M(self, cmap: str = 'PuBuGn') -> None:
        """
        Plot the marginal means produced by the generative algorithm.

        Parameters
        ----------
        cmap : Matplotlib object
               Colormap used for the plot.
        """

        M = (self.M0 + self.eta * self.M0 * transpose_tensor(self.M0)) / self.Z
        for l in range(self.L):
            np.fill_diagonal(M[l], 0.)
            _, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(M[l], cmap=plt.get_cmap(cmap))
            ax.set_title(f'Marginal means matrix layer {l}', fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    # pylint: enable=W0631
