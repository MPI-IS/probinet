from abc import ABCMeta
import logging
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import brentq

from ..input.stats import print_graph_stat
from ..input.tools import (
    check_symmetric, Exp_ija_matrix, log_and_raise_error, normalize_nonzero_membership,
    output_adjacency, transpose_tensor)
from ..output.evaluate import lambda_full
from ..output.plot import plot_A

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
        **kwargs,  # these kwargs are needed later on
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
