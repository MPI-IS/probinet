"""
Code to generate multilayer networks with non-negative and discrete weights, and whose nodes are associated
with one categorical attribute. Self-loops are removed and only the largest connected component is considered.
"""

import logging
import os
import warnings
from abc import ABCMeta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from probinet.evaluation.expectation_computation import compute_mean_lambda0
from probinet.input.stats import print_graph_stats
from probinet.models.constants import OUTPUT_FOLDER
from probinet.synthetic.base import StandardMMSBM
from probinet.utils.matrix_operations import normalize_nonzero_membership
from probinet.utils.tools import get_or_create_rng, output_adjacency
from probinet.visualization.plot import plot_A

DEFAULT_N = 100
DEFAULT_L = 1
DEFAULT_K = 2
DEFAULT_Z = 2
DEFAULT_AVG_DEGREE = 15

DEFAULT_DIRECTED = False
DEFAULT_PERC_OVERLAPPING = 0.4
DEFAULT_CORRELATION_U_V = 0.0
DEFAULT_ALPHA = 0.1

DEFAULT_STRUCTURE = "assortative"

DEFAULT_SEED = 0

DEFAULT_IS_SPARSE = True

DEFAULT_OUT_FOLDER = "data/input/synthetic/"

DEFAULT_SHOW_DETAILS = False
DEFAULT_SHOW_PLOTS = False
DEFAULT_OUTPUT_NET = False


def pi_ik_matrix(u: np.ndarray, v: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute the mean pi_ik for all entries.

    Parameters
    ----------
    u : np.ndarray
        Out-going membership matrix.
    v : np.ndarray
        In-coming membership matrix.
    beta : np.ndarray
        Affinity matrix.

    Returns
    -------
    pi : np.ndarray
    """

    pi = 0.5 * np.matmul((u + v), beta)

    return pi


def output_design_matrix(X, out_folder, label):
    """
    Save the design matrix tensor to a file.

    INPUT
    ----------
    X : np.ndarray
        One-hot encoding of design matrix.
    out_folder : str
                 Path to store the design matrix.
    label : str
            Label name to store the design matrix.
    """

    outfile = label + ".csv"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    df = pd.DataFrame(X, columns=["Metadata"])
    df["nodeID"] = df.index
    df = df.loc[:, ["nodeID", "Metadata"]]
    df.to_csv(out_folder + outfile, index=False, sep=" ")

    print(f"Design matrix saved in: {out_folder + outfile}")


def plot_X(X, cmap="PuBuGn"):
    """
    Plot the design matrix produced by the generative algorithm.

    INPUT
    ----------
    X : np.ndarray
        Design matrix.
    cmap : Matplotlib object
           Colormap used for the plot.
    """

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(pd.get_dummies(X), cmap=plt.get_cmap(cmap), aspect="auto")
    ax.set_title("Design matrix", fontsize=15)
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)
    plt.show()


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
        Z: int = DEFAULT_Z,
        out_folder: Path = OUTPUT_FOLDER,
        output_net: bool = DEFAULT_OUTPUT_NET,
        show_details: bool = DEFAULT_SHOW_DETAILS,
        show_plots: bool = DEFAULT_SHOW_PLOTS,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities
        self.Z = Z  # number of categories of the categorical attribute

        # Set  random number generator
        self.rng = get_or_create_rng(rng)

        self.out_folder = out_folder
        self.output_data = output_net

        self.show_details = show_details
        self.show_plots = show_plots


class SyntheticMTCOV(BaseSyntheticNetwork, StandardMMSBM):
    """
    Create a synthetic, possibly directed, and weighted network (possibly multilayer)
    by a standard mixed-membership stochastic block-model
    - It models marginals (iid assumption) with Poisson distributions
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        parameters = kwargs.get("parameters", None)
        attributes = kwargs.get("attributes", None)

        self.init_mmsbm_params(**kwargs)

        self.build_Y(parameters=parameters)

        self.build_X(attributes=attributes)

        if self.output_data:
            self._output_parameters()
            output_adjacency(self.layer_graphs, self.out_folder, "A" + self.label)
            output_design_matrix(self.X, self.out_folder, "X" + self.label)

        if self.show_details:
            print_graph_stats(self.G)
        if self.show_plots:
            plot_A(self.layer_graphs)
            plot_X(self.X)
            if self.M is not None:
                self._plot_M()
            if self.pi is not None:
                self._plot_pi()

    def init_mmsbm_params(self, **kwargs):
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

        if "directed" in kwargs:
            directed = kwargs["directed"]
        else:
            msg = f"directed parameter was not set. Defaulting to directed={DEFAULT_DIRECTED}"
            warnings.warn(msg)
            directed = DEFAULT_DIRECTED
        self.directed = directed

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
            msg = "label parameter was not set. Defaulting to label=_N_L_K_avgdegree"
            warnings.warn(msg)
            label = "_".join(
                [
                    str(),
                    str(self.N),
                    str(self.L),
                    str(self.K),
                    str(self.avg_degree),
                ]
            )
        self.label = label

        # SETUP overlapping communities

        if "perc_overlapping" in kwargs:
            perc_overlapping = kwargs["perc_overlapping"]
            if (perc_overlapping < 0) or (
                perc_overlapping > 1
            ):  # fraction of nodes with mixed membership
                err_msg = "The percentage of overlapping nodes has to be in [0, 1]!"
                raise ValueError(err_msg)
        else:
            msg = f"perc_overlapping parameter was not set. Defaulting to perc_overlapping={DEFAULT_PERC_OVERLAPPING}"
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
                msg = (
                    f"correlation_u_v parameter for overlapping communities was not set. "
                    f"Defaulting to corr={DEFAULT_CORRELATION_U_V}"
                )
                warnings.warn(msg)
                correlation_u_v = DEFAULT_CORRELATION_U_V
            self.correlation_u_v = correlation_u_v

            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            else:
                msg = f"alpha parameter of Dirichlet distribution was not set. Defaulting to alpha={[DEFAULT_ALPHA] * self.K}"
                warnings.warn(msg)
                alpha = [DEFAULT_ALPHA] * self.K
            if isinstance(alpha, float):
                if alpha <= 0:
                    err_msg = (
                        "Each entry of the Dirichlet parameter has to be positive!"
                    )
                    raise ValueError(err_msg)
                else:
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
            msg = f"structure parameter was not set. Defaulting to structure={[DEFAULT_STRUCTURE] * self.L}"
            warnings.warn(msg)
            structure = [DEFAULT_STRUCTURE] * self.L
        if isinstance(structure, str):
            if structure not in ["assortative", "disassortative"]:
                err_msg = "The available structures for the affinity tensor w are: assortative, disassortative!"
                raise ValueError(err_msg)
            else:
                structure = [structure] * self.L
        elif len(structure) != self.L:
            err_msg = (
                "The parameter structure should be a list of length L. "
                "Each entry defines the structure of the corresponding layer!"
            )
            raise ValueError(err_msg)
        for e in structure:
            if e not in ["assortative", "disassortative"]:
                err_msg = "The available structures for the affinity tensor w are: assortative, disassortative!"
                raise ValueError(err_msg)
        self.structure = structure

    def build_Y(self, parameters=None):
        """
        Generate network layers G using the latent variables,
        with the generative model A_ij ~ P(A_ij|u,v,w)
        """

        # Latent variables

        if parameters is None:
            # generate latent variables
            self.u, self.v, self.w = self._generate_lv()
        else:
            # set latent variables
            self.u, self.v, self.w = parameters
            if self.u.shape != (self.N, self.K):
                raise ValueError("The shape of the parameter u has to be (N, K).")
            if self.v.shape != (self.N, self.K):
                raise ValueError("The shape of the parameter v has to be (N, K).")
            if self.w.shape != (self.L, self.K, self.K):
                raise ValueError("The shape of the parameter w has to be (L, K, K).")
            self.u = normalize_nonzero_membership(self.u, axis=1)
            self.v = normalize_nonzero_membership(self.v, axis=1)

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
        if not self.directed:
            # symmetrize
            for layer in range(self.L):
                Y[layer] = Y[layer] + Y[layer].T - np.diag(Y[layer].diagonal())

        # Create networkx Graph objects for each layer for easier manipulation

        nodes_to_remove = []
        self.G = []
        self.layer_graphs = []
        for layer in range(self.L):
            if self.directed:
                self.G.append(nx.from_numpy_array(Y[layer], create_using=nx.DiGraph()))
                Gc = max(nx.weakly_connected_components(self.G[layer]), key=len)
                nodes_to_remove.append(set(self.G[layer].nodes()).difference(Gc))
            else:
                self.G.append(nx.from_numpy_array(Y[layer], create_using=nx.Graph()))

        if self.directed:
            n_to_remove = nodes_to_remove[0].intersection(*nodes_to_remove)
        for layer in range(self.L):
            if self.directed:
                self.G[layer].remove_nodes_from(list(n_to_remove))
            self.nodes = list(self.G[layer].nodes())

            self.layer_graphs.append(
                nx.to_scipy_sparse_array(self.G[layer], nodelist=self.nodes)
            )

        self.u = self.u[self.nodes]
        self.v = self.v[self.nodes]
        self.N = len(self.nodes)
        self.Y = Y[np.ix_(np.arange(self.L), self.nodes, self.nodes)]

    def build_X(self, attributes: Optional[np.ndarray] = None):
        """

        Generate the design matrix.

        Parameters
        ----------
        attributes : np.ndarray, optional
            The latent variables representing the contribution of the attributes.
            If None, the attributes will be generated.

        Raises
        ------
        ValueError
                If the shape of the parameter `beta` is not (K, Z).
        """
        # Latent variables

        if attributes is None:
            # generate attributes
            self.beta = self._generate_lv_attributes()
        else:
            # set attributes
            self.beta = attributes
            if self.beta.shape != (self.K, self.Z):
                raise ValueError("The shape of the parameter beta has to be (K, Z).")
            self.beta = normalize_nonzero_membership(self.beta, axis=1)

        # Generate X

        self.pi = pi_ik_matrix(self.u, self.v, self.beta)
        categories = [f"attr{z+1}" for z in range(self.Z)]
        self.X = np.array(
            [self.rng.choice(categories, p=self.pi[i]) for i in np.arange(self.N)]
        ).reshape(self.N)
        try:
            self.X = self.X[self.nodes]
        except IndexError:
            logging.debug("X couldn't be sliced by nodes.")

    def _generate_lv_attributes(self):
        """
        Generate latent variables representing the contribution of the attributes.
        """
        # Generate beta
        beta = np.zeros((self.K, self.Z))
        for k in range(min(self.K, self.Z)):
            beta[k, k] = 1.0
        if self.Z > self.K:
            for z in range(self.K, self.Z):
                beta[:, z] = 0.5
        if self.Z < self.K:
            for k in range(self.Z, self.K):
                beta[k, :] = 0.5

        return beta

    def _plot_M(self, cmap="PuBuGn"):
        """
        Plot the marginal means produced by the generative algorithm.
        """

        for layer in range(self.L):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.matshow(self.M[layer], cmap=plt.get_cmap(cmap))
            ax.set_title(f"Marginal means matrix layer {layer}", fontsize=15)
            for PCM in ax.get_children():
                if isinstance(PCM, plt.cm.ScalarMappable):
                    break
            plt.colorbar(PCM, ax=ax)
            plt.show()

    def _plot_pi(self, cmap="PuBuGn"):
        """
        Plot the parameters of the categorical attribute produced by the generative algorithm.
        """

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.pi, cmap=plt.get_cmap(cmap), aspect="auto")
        ax.set_title(f"Marginal means categorical attribute", fontsize=15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()
