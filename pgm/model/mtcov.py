"""
Class definition of MTCOV, the generative algorithm that incorporates both the topology of interactions and node
attributes to extract overlapping communities in directed and undirected multilayer networks.
"""

import logging
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import sktensor as skt
from typing_extensions import Unpack

from ..input.preprocessing import preprocess, preprocess_X
from ..input.tools import sp_uttkrp, sp_uttkrp_assortative
from ..output.evaluate import lambda_full
from .base import ModelBase, ModelFitParameters, ModelUpdateMixin
from .constants import CONVERGENCE_TOL_, DECISION_, ERR_, INF_


class MTCOV(ModelBase, ModelUpdateMixin):
    """
    Class definition of MTCOV, the generative algorithm that incorporates both the topology of interactions and
    node attributes to extract overlapping communities in directed and undirected multilayer networks.
    """

    def __init__(
        self,
        inf: float = INF_,  # initial value of the log-likelihood
        err_max: float = 0.0000001,  # minimum value for the parameters
        err: float = ERR_,  # noise for the initialization
        num_realizations: int = 1,  # number of iterations with different random initialization
        convergence_tol: float = CONVERGENCE_TOL_,  # tolerance for convergence
        decision: int = DECISION_,  # convergence parameter
        max_iter: int = 500,  # maximum number of EM steps before aborting
        plot_loglik: bool = False,  # flag to plot the log-likelihood
        flag_conv: str = "log",  # flag to choose the convergence criterion
    ) -> None:

        super().__init__(
            inf,
            err_max,
            err,
            num_realizations,
            convergence_tol,
            decision,
            max_iter,
            plot_loglik,
            flag_conv,
        )

        # Initialize the attributes
        self.u_f: np.ndarray = np.array([])
        self.v_f: np.ndarray = np.array([])
        self.w_f: np.ndarray = np.array([])

    def check_fit_params(
        self,
        initialization: int,
        gamma: float,
        undirected: bool,
        assortative: bool,
        data: Union[skt.dtensor, skt.sptensor, np.ndarray],
        data_X: Union[skt.dtensor, skt.sptensor, np.ndarray],
        K: int,
        **extra_params: Unpack[ModelFitParameters],
    ) -> None:

        message = (
            "The initialization parameter can be either 0 or 1. It is used as an indicator to "
            "initialize the membership matrices u and v and the affinity matrix w. If it is 0, they "
            "will be generated randomly, otherwise they will upload from file."
        )
        available_extra_params = [
            "files",
            "out_inference",
            "out_folder",
            "end_file",
        ]

        super()._check_fit_params(
            initialization,
            undirected,
            assortative,
            data,
            K,
            available_extra_params,
            data_X,
            eta0=None,
            beta0=None,
            gamma=gamma,
            message=message,
            **extra_params,
        )

        # Parameters for the initialization of the model
        self.normalize_rows = False

        if self.initialization == 1:
            self.theta = np.load(self.files, allow_pickle=True)
            dfW = self.theta["w"]
            self.L = dfW.shape[0]
            self.K = dfW.shape[1]
            dfU = self.theta["u"]
            self.N = dfU.shape[0]
            dfB = self.theta["beta"]
            self.Z = dfB.shape[1]
            assert self.K == dfU.shape[1] == dfB.shape[0]

    def preprocess_data_for_fit(
        self,
        data: Union[skt.dtensor, skt.sptensor],
        data_X: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[
        Union[skt.dtensor, skt.sptensor],
        np.ndarray,
        Tuple[np.ndarray],
        Tuple[np.ndarray],
        Optional[np.ndarray],
        Optional[Tuple[np.ndarray]],
        Optional[Tuple[np.ndarray]],
    ]:
        """
        Preprocesses the input data for fitting the model.

        This method handles the sparsity of the data, saves the indices of the non-zero entries,
        and optionally selects a subset of nodes for batch processing.

        Parameters
        ----------
        data : Union[skt.dtensor, skt.sptensor]
            The graph adjacency tensor to be preprocessed.
        data_X : np.ndarray
            The one-hot encoding version of the design matrix to be preprocessed.
        batch_size : Optional[int], default=None
            The size of the subset of nodes to compute the likelihood with. If None, the method
            will automatically determine the batch size based on the number of nodes.

        Returns
        -------
        preprocessed_data : Union[skt.dtensor, skt.sptensor]
            The preprocessed graph adjacency tensor.
        preprocessed_data_X : np.ndarray
            The preprocessed one-hot encoding version of the design matrix.
        subs_nz : Tuple[np.ndarray]
            The indices of the non-zero entries in the data.
        subs_X_nz : Tuple[np.ndarray]
            The indices of the non-zero entries in the design matrix.
        subset_N : Optional[np.ndarray]
            The subset of nodes selected for batch processing. None if no subset is selected.
        Subs : Optional[Tuple[np.ndarray]]
            The list of tuples representing the non-zero entries in the data. None if no subset is selected.
        SubsX : Optional[Tuple[np.ndarray]]
            The list of tuples representing the non-zero entries in the design matrix. None if no subset is selected.
        """

        # Pre-processing of the data to handle the sparsity
        if not isinstance(data, skt.sptensor):
            data = preprocess(data)
        data_X = preprocess_X(data_X)

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs
        subs_X_nz = data_X.nonzero()

        if batch_size:
            if batch_size > self.N:
                batch_size = min(5000, self.N)
            np.random.seed(10)  # TODO: ask Martina why this seed
            subset_N = np.random.choice(
                np.arange(self.N), size=batch_size, replace=False
            )
            Subs = list(zip(*subs_nz))
            SubsX = list(zip(*subs_X_nz))
        else:
            if self.N > 5000:
                batch_size = 5000
                np.random.seed(10)
                subset_N = np.random.choice(
                    np.arange(self.N), size=batch_size, replace=False
                )
                Subs = list(zip(*subs_nz))
                SubsX = list(zip(*subs_X_nz))
            else:
                subset_N = None
                Subs = None
                SubsX = None
        logging.debug("batch_size: %s", batch_size)

        return data, data_X, subs_nz, subs_X_nz, subset_N, Subs, SubsX  # type: ignore

    def fit(
        self,
        data: Union[skt.dtensor, skt.sptensor],
        data_X: np.ndarray,
        nodes: List[Any],
        flag_conv: str = "log",
        batch_size: Optional[int] = None,
        gamma: float = 0.5,
        rseed: int = 0,
        K: int = 3,
        initialization: int = 0,
        undirected: bool = False,
        assortative: bool = True,
        **extra_params: Unpack[ModelFitParameters],
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        float,
    ]:
        """
        Performing community detection in multilayer networks considering both the topology of interactions and node
        attributes via EM updates.
        Save the membership matrices U and V, the affinity tensor W and the beta matrix.

        Parameters
        ----------
        data : ndarray/sptensor
               Graph adjacency tensor.
        data_X : ndarray
                 Object representing the one-hot encoding version of the design matrix.
        flag_conv : str
                    If 'log' the convergence is based on the loglikelihood values; if 'deltas' the convergence is
                    based on the differences in the parameters values. The latter is suggested when the dataset
                    is big (N > 1000 ca.).
        nodes : list
                List of nodes IDs.
        batch_size : int/None
                     Size of the subset of nodes to compute the likelihood with.
        gamma : float
                Weight of the node attributes.
        rseed : int
                Random seed.
        K : int
            Number of communities.
        initialization : int
                        If 0, the membership matrices u and v and the affinity matrix w are generated randomly;
                        if 1, they are uploaded from file.
        undirected : bool
                        If True, the model is undirected.
        assortative : bool
                        If True, the model is assortative.
        extra_params : dict
                        Additional parameters.
        Returns
        -------
        u_f : ndarray
              Membership matrix (out-degree).
        v_f : ndarray
              Membership matrix (in-degree).
        w_f : ndarray
              Affinity tensor.
        beta_f : ndarray
                 Beta parameter matrix.
        maxL : float
               Maximum log-likelihood value.
        """
        # Check the parameters for fitting the model
        self.check_fit_params(
            data=data,
            data_X=data_X,
            K=K,
            initialization=initialization,
            gamma=gamma,
            undirected=undirected,
            assortative=assortative,
            **extra_params,
        )
        # Set the random seed
        logging.debug("Fixing random seed to: %s", rseed)
        self.rng = np.random.RandomState(rseed)  # pylint: disable=no-member

        # Initialize the fit parameters
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum log-likelihood
        self.nodes = nodes

        # Preprocess the data for fitting the model
        data, data_X, subs_nz, subs_X_nz, subset_N, Subs, SubsX = (
            self.preprocess_data_for_fit(data, data_X, batch_size)
        )

        # Set the preprocessed data and other related variables as attributes of the class instance
        self.data = data
        self.data_X = data_X
        self.subs_nz = subs_nz
        self.subs_X_nz = subs_X_nz
        self.batch_size = batch_size
        self.subset_N = subset_N
        self.Subs = Subs
        self.SubsX = SubsX

        # The following part of the code is responsible for running the Expectation-Maximization
        # (EM)  algorithm for a specified number of realizations (self.num_realizations):
        for r in range(self.num_realizations):

            # Initialize the parameters for the current realization
            coincide, convergence, it, loglik, loglik_values = (
                self._initialize_realization()
            )

            # Update the parameters for the current realization
            it, loglik, coincide, convergence, loglik_values = self._update_realization(
                r, it, loglik, coincide, convergence, loglik_values
            )

            # If the current log-likelihood is greater than the maximum log-likelihood so far,
            # update the optimal parameters and the maximum log-likelihood
            if maxL < loglik:
                super()._update_optimal_parameters()
                self.maxL = loglik
                self.final_it = it
                conv = convergence
                self.best_r = r
                if flag_conv == "log":
                    best_loglik_values = list(loglik_values)
            # Log the current realization number, log-likelihood, number of iterations, and elapsed time
            self._log_realization_info(
                r, loglik, self.final_it, self.time_start, convergence
            )

        # End cycle over realizations

        # Evaluate the results of the fitting process
        self._evaluate_fit_results(self.maxL, conv, best_loglik_values)

        # Return the final parameters and the maximum log-likelihood
        return self.u_f, self.v_f, self.w_f, self.beta_f, self.maxL

    def _initialize_realization(self):
        """
        This method initializes the parameters, updates the old variables
        and updates the cache.  It sets up local variables for convergence checking.
        coincide and it are counters, convergence is a boolean flag, and loglik is the
        initial log-likelihood.
        """
        # Log the current state of the random number generator
        logging.debug("Random number generator seed: %s", self.rng.get_state()[1][0])

        # Call the _initialize method from the parent class to initialize the parameters for the current realization
        super()._initialize()

        # Call the _update_old_variables method from the parent class to update the old variables for the current realization
        super()._update_old_variables()

        # Update the cache used in the EM update
        self._update_cache(self.data, self.subs_nz, self.data_X, self.subs_X_nz)  # type: ignore

        # Set up local variables for convergence checking
        # coincide and it are counters, convergence is a boolean flag
        # loglik is the initial pseudo log-likelihood
        coincide, it = 0, 0
        convergence = False
        loglik = self.inf
        loglik_values = []

        # Record the start time of the realization
        self.time_start = time.time()

        # Return the initial state of the realization
        return coincide, convergence, it, loglik, loglik_values

    def compute_likelihood(self):
        """
        Compute the pseudo log-likelihood of the data.

        Returns
        -------
        loglik : float
                 Pseudo log-likelihood value.
        """
        if not self.batch_size:
            return self._likelihood()
        else:
            return self._likelihood_batch(
                self.data, self.data_X, self.subset_N, self.Subs, self.SubsX
            )

    def _update_em(self):
        """
        Update parameters via EM procedure.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_X : ndarray
                 Object representing the one-hot encoding version of the design matrix.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        subs_X_nz : tuple
                    Indices of elements of data_X that are non-zero.
        """

        if self.gamma < 1.0:
            if not self.assortative:
                d_w = self._update_W()
            else:
                d_w = self._update_W_assortative()
        else:
            d_w = 0
        self._update_cache(self.data, self.subs_nz, self.data_X, self.subs_X_nz)

        if self.gamma > 0.0:
            d_beta = self._update_beta(self.subs_X_nz)
        else:
            d_beta = 0.0
        self._update_cache(self.data, self.subs_nz, self.data_X, self.subs_X_nz)

        d_u = self._update_U()
        self._update_cache(self.data, self.subs_nz, self.data_X, self.subs_X_nz)

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
        else:
            d_v = self._update_V()
        self._update_cache(self.data, self.subs_nz, self.data_X, self.subs_X_nz)

        self.delta_u = d_u
        self.delta_v = d_v
        self.delta_w = d_w
        self.delta_beta = d_beta

    def _initialize_eta(self) -> None:
        """
        Override the _initialize_eta method in MTCOV class to do nothing.
        """

    def _file_initialization(self) -> None:
        # Call the _file_initialization method of the parent class
        super()._file_initialization()
        # Initialiue beta from file
        self._initialize_beta_from_file()

    def _random_initialization(self) -> None:
        # Log a message indicating that u, v and w are being initialized randomly
        logging.debug("%s", "u, v and w are initialized randomly.")

        # Randomize u, v
        self._randomize_u_v(normalize_rows=self.normalize_rows)
        # If gamma is not 0, randomize beta matrix
        if self.gamma != 0:
            self._randomize_beta(
                (self.K, self.Z)
            )  # Generates a matrix of random numbers
        # If gamma is not 1, randomize w
        if self.gamma != 1:
            self._randomize_w()

    def _update_cache(
        self,
        data: Union[skt.dtensor, skt.sptensor],
        subs_nz: Tuple[np.ndarray],
        data_X: np.ndarray,
        subs_X_nz: Tuple[np.ndarray],
    ) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : Union[skt.dtensor, skt.sptensor]
               Graph adjacency tensor.
        subs_nz : Tuple[np.ndarray]
                  Indices of elements of data that are non-zero.
        data_X : np.ndarray
                 Object representing the one-hot encoding version of the design matrix.
        subs_X_nz : Tuple[np.ndarray]
                    Indices of elements of data_X that are non-zero.
        """

        # A
        self.lambda0_nz = super()._lambda_nz(subs_nz)
        self.lambda0_nz[self.lambda0_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.lambda0_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.lambda0_nz

        # X
        self.pi0_nz = self._pi0_nz(subs_X_nz, self.u, self.v, self.beta)
        self.pi0_nz[self.pi0_nz == 0] = 1
        if not scipy.sparse.issparse(data_X):
            self.data_pi_nz = data_X[subs_X_nz[0]] / self.pi0_nz
        else:
            self.data_pi_nz = data_X.data / self.pi0_nz

    def _pi0_nz(
        self,
        subs_X_nz: Tuple[np.ndarray],
        u: np.ndarray,
        v: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the mean pi0 (pi_iz) for only non-zero entries (denominator of hizk).

        Parameters
        ----------
        subs_X_nz : tuple
                    Indices of elements of data_X that are non-zero.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        beta : ndarray
               Beta matrix.

        Returns
        -------
        Mean pi0 (pi_iz) for only non-zero entries.
        """

        if self.undirected:
            return np.einsum("Ik,kz->Iz", u[subs_X_nz[0], :], beta)
        return np.einsum("Ik,kz->Iz", u[subs_X_nz[0], :] + v[subs_X_nz[0], :], beta)

    def _specific_update_W(self):

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum(
            "Ik,Iq->Ikq", self.u[self.subs_nz[1], :], self.v[self.subs_nz[2], :]
        )
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV

        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(
                    self.subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L
                )
        self.w *= uttkrp_DKQ
        Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))
        non_zeros = Z > 0

        self.w[:, non_zeros] /= Z[non_zeros]

    def _specific_update_W_assortative(self):
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum(
            "Ik,Ik->Ik", self.u[self.subs_nz[1], :], self.v[self.subs_nz[2], :]
        )
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV

        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(
                self.subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L
            )

        self.w *= uttkrp_DKQ

        Z = (self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0))
        non_zeros = Z > 0
        for a in range(self.L):
            self.w[a, non_zeros] /= Z[non_zeros]

    def _update_beta(self, subs_X_nz: Tuple[np.ndarray]) -> float:
        """
        Update beta matrix.

        Parameters
        ----------
        subs_X_nz : tuple
                    Indices of elements of data_X that are non-zero.

        Returns
        -------
        dist_beta : float
                    Maximum distance between the old and the new beta matrix.
        """

        if self.undirected:
            XUV = np.einsum("Iz,Ik->kz", self.data_pi_nz, self.u[subs_X_nz[0], :])
        else:
            XUV = np.einsum(
                "Iz,Ik->kz",
                self.data_pi_nz,
                self.u[subs_X_nz[0], :] + self.v[subs_X_nz[0], :],
            )
        self.beta *= XUV

        row_sums = self.beta.sum(axis=1)
        self.beta[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.beta < self.err_max  # values are too low
        self.beta[low_values_indices] = 0.0  # and set to 0.

        dist_beta = np.amax(abs(self.beta - self.beta_old))  # type: ignore
        self.beta_old = np.copy(self.beta)

        return dist_beta

    def _specific_update_U(self) -> None:
        self.u = self._update_membership(
            self.subs_nz, self.subs_X_nz, self.u, self.v, self.w, self.beta, 1
        )

        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _specific_update_V(self) -> None:
        self.v = self._update_membership(
            self.subs_nz, self.subs_X_nz, self.u, self.v, self.w, self.beta, 2
        )

        row_sums = self.v.sum(axis=1)
        self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _update_membership(
        self,
        subs_nz: Tuple[np.ndarray],
        subs_X_nz: Tuple[np.ndarray],
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        beta: np.ndarray,
        m: int,
    ) -> np.ndarray:
        """
        Main procedure to update membership matrices.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        subs_X_nz : tuple
                    Indices of elements of data_X that are non-zero.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        beta : ndarray
               Beta matrix.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix U; if 2 it works with V.

        Returns
        -------
        out : ndarray
              Update of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m, u, v, w)

        if m == 1:
            uttkrp_DK *= u
        elif m == 2:
            uttkrp_DK *= v

        uttkrp_Xh = np.einsum("Iz,kz->Ik", self.data_pi_nz, beta)

        if self.undirected:
            uttkrp_Xh *= u[subs_X_nz[0]]
        else:
            uttkrp_Xh *= u[subs_X_nz[0]] + v[subs_X_nz[0]]

        uttkrp_DK *= 1 - self.gamma  # type: ignore
        out = uttkrp_DK.copy()
        out[subs_X_nz[0]] += self.gamma * uttkrp_Xh

        return out

    def _likelihood(self) -> float:
        """
        Compute the log-likelihood of the data.

        Returns
        -------
        l : float
            Log-likelihood value.
        """

        self.lambda0_ija = lambda_full(self.u, self.v, self.w)
        lG = -self.lambda0_ija.sum()
        logM = np.log(self.lambda0_nz)
        if isinstance(self.data, skt.dtensor):
            Alog = self.data[self.data.nonzero()] * logM
        elif isinstance(self.data, skt.sptensor):
            Alog = self.data.vals * logM
        lG += Alog.sum()

        if self.undirected:
            logP = np.log(self.pi0_nz)
        else:
            logP = np.log(0.5 * self.pi0_nz)
        if not scipy.sparse.issparse(self.data_X):
            ind_logP_nz = (np.arange(len(logP)), self.data_X.nonzero()[1])
            Xlog = self.data_X[self.data_X.nonzero()] * logP[ind_logP_nz]
        else:
            Xlog = self.data_X.data * logP
        lX = Xlog.sum()

        loglik = (1.0 - self.gamma) * lG + self.gamma * lX  # type: ignore

        if np.isnan(loglik):
            raise ValueError("Likelihood is NaN!!!!")

        return loglik

    def _likelihood_batch(
        self,
        data: Union[skt.dtensor, skt.sptensor],
        data_X: np.ndarray,
        subset_N: List[int],
        Subs: List[Tuple[int, int, int]],
        SubsX: List[Tuple[int, int]],
    ) -> float:
        """
        Compute the log-likelihood of a batch of data.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_X : ndarray
                 Object representing the one-hot encoding version of the design matrix.
        subset_N : list
                   List with a subset of nodes.
        Subs : list
               List with elements (a, i, j) of the non zero entries of data.
        SubsX : list
                List with elements (i, z) of the non zero entries of data_X.

        Returns
        -------
        l : float
            Log-likelihood value.
        """

        size = len(subset_N)
        self.lambda0_ija = lambda_full(self.u, self.v, self.w)
        assert self.lambda0_ija.shape == (self.L, size, size)
        lG = -self.lambda0_ija.sum()
        logM = np.log(self.lambda0_nz)
        IDXs = [
            i for i, e in enumerate(Subs) if (e[1] in subset_N) and (e[2] in subset_N)
        ]
        Alog = data.vals[IDXs] * logM[IDXs]
        lG += Alog.sum()

        if self.undirected:
            logP = np.log(self.pi0_nz)
        else:
            logP = np.log(0.5 * self.pi0_nz)
        if size:
            IDXs = [i for i, e in enumerate(SubsX) if (e[0] in subset_N)]
        else:
            IDXs = []
        X_attr = scipy.sparse.csr_matrix(data_X)
        Xlog = X_attr.data[IDXs] * logP[(IDXs, X_attr.nonzero()[1][IDXs])]
        lX = Xlog.sum()

        loglik = (1.0 - self.gamma) * lG + self.gamma * lX  # type: ignore

        if np.isnan(loglik):
            logging.error("Likelihood is NaN!!!!")
            sys.exit(1)
        else:
            return loglik

    def _check_for_convergence_delta(
        self, it, coincide, du, dv, dw, db, convergence
    ):  # pylint: disable=arguments-renamed
        """
        Check for convergence by using the maximum distances between the old and the new parameters values.

        Parameters
        ----------
        it : int
             Number of iteration.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        du : float
             Maximum distance between the old and the new membership matrix U.
        dv : float
             Maximum distance between the old and the new membership matrix V.
        dw : float
             Maximum distance between the old and the new affinity tensor W.
        db : float
             Maximum distance between the old and the new beta matrix.
        convergence : bool
                      Flag for convergence.

        Returns
        -------
        it : int
             Number of iteration.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        """

        if (
            du < self.convergence_tol
            and dv < self.convergence_tol
            and dw < self.convergence_tol
            and db < self.convergence_tol
        ):
            coincide += 1
        else:
            coincide = 0

        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _copy_variables(self, source_suffix: str, target_suffix: str) -> None:
        """
        Copy variables from source to target.

        Parameters
        ----------
        source_suffix : str
                        The suffix of the source variable names.
        target_suffix : str
                        The suffix of the target variable names.
        """
        # Call the base method
        super()._copy_variables(source_suffix, target_suffix)

        # Set specific variables
        source_var = getattr(self, f"beta{source_suffix}")
        setattr(self, f"beta{target_suffix}", np.copy(source_var))
