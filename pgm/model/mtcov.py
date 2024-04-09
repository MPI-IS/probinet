"""
Class definition of MTCov, the generative algorithm that incorporates both the topology of interactions and node
attributes to extract overlapping communities in directed and undirected multilayer networks.
"""

from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
import scipy.sparse
import sktensor as skt
from termcolor import colored
from typing_extensions import Unpack

from ..input.preprocessing import preprocess, preprocess_X
from ..input.tools import sp_uttkrp, sp_uttkrp_assortative
from ..model.crep import FitParams
from ..output.plot import plot_L


class MTCov:
    """
    Class definition of MTCov, the generative algorithm that incorporates both the topology of interactions and
    node attributes to extract overlapping communities in directed and undirected multilayer networks.
    """

    def __init__(self,
                 inf: float = 1e10,
                 err_max: float = 0.0000001,
                 err: float = 0.1,
                 num_realizations: int = 1,
                 convergence_tol: float = 0.0001,
                 decision: int = 10,
                 max_iter: int = 500,
                 verbose: bool = False,
                 plot_loglik: bool = False,  # flag to plot the log-likelihood
                 flag_conv: str = 'log'  # flag to choose the convergence criterion
                 ) -> None:

        self.inf = inf  # initial value of the pseudo log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.num_realizations = num_realizations  # number of iterations with different random initialization
        self.convergence_tol = convergence_tol  # convergence_tol parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.verbose = verbose  # print verbose
        self.plot_loglik = plot_loglik  # flag to plot the log-likelihood
        self.flag_conv = flag_conv

    def __check_fit_parameters(self,
                               initialization: int,
                               gamma: float,
                               undirected: bool,
                               assortative: bool,
                               data: Union[skt.dtensor, skt.sptensor, np.ndarray],
                               data_X: Union[skt.dtensor, skt.sptensor, np.ndarray],
                               K: int,
                               **extra_params: Unpack[FitParams]
                               ) -> None:

        if "files" in extra_params:
            self.files = extra_params["files"]
        else:
            raise ValueError('The input file is missing.')

        if initialization not in {0, 1}:  # indicator for choosing how to initialize u, v and w
            raise ValueError(
                'The initialization parameter can be either 0 or 1. It is used as an indicator to '
                'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
                'will be generated randomly, otherwise they will upload from file.')
        self.initialization = initialization

        if self.initialization == 1:
            self.theta = np.load(self.files, allow_pickle=True)
            dfW = self.theta['w']
            self.L = dfW.shape[0]
            self.K = dfW.shape[1]
            dfU = self.theta['u']
            self.N = dfU.shape[0]
            dfB = self.theta['beta']
            self.Z = dfB.shape[1]
            assert self.K == dfU.shape[1] == dfB.shape[0]

        self.gamma = gamma  # scaling parameter
        self.undirected = undirected  # flag for undirected networks
        self.assortative = assortative

        self.N = data.shape[1]
        self.L = data.shape[0]
        self.K = K  # number of communities
        self.Z = data_X.shape[1]  # number of categories of the categorical attribute

        available_extra_params = [
            'files',
            'out_inference',
            'out_folder',
            'end_file',
        ]

        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = f'Ignoring extra parameter {extra_param}.'
                print(msg)  # Add the warning

        if "out_inference" in extra_params:
            self.out_inference = extra_params["out_inference"]
        else:
            self.out_inference = True
        if "out_folder" in extra_params:
            self.out_folder = extra_params["out_folder"]
        else:
            self.out_folder = Path('outputs')

        if "end_file" in extra_params:
            self.end_file = extra_params["end_file"]
        else:
            self.end_file = ''

        # values of the parameters used during the update
        self.beta = np.zeros((self.K, self.Z), dtype=float)

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

    def fit(self,
            data: Union[skt.dtensor, skt.sptensor],
            data_X: np.ndarray,
            nodes: List[Any],
            flag_conv: str = 'log',
            batch_size: Optional[int]= None,
            gamma: float = 0.5,
            rseed: int = 0,
            K: int = 3,
            initialization: int = 0,
            undirected: bool = False,
            assortative: bool = True,
            **extra_params: Unpack[FitParams]  # TODO: could this be done in another way? mypy keeps
            # complaining about the types of the values
            ) -> tuple[np.ndarray[Any,
                                  np.dtype[np.float64]],
                       np.ndarray[Any,
                                  np.dtype[np.float64]],
                       np.ndarray[Any,
                                  np.dtype[np.float64]],
                       np.ndarray[Any,
                                  np.dtype[np.float64]],
                       float]:
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
        self.__check_fit_parameters(data=data,
                                    data_X=data_X,
                                    K=K,
                                    initialization=initialization,
                                    gamma=gamma,
                                    undirected=undirected,
                                    assortative=assortative,
                                    **extra_params
                                    )

        self.rseed = rseed
        self.rng = np.random.RandomState(self.rseed)  # pylint: disable=no-member
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood

        # pre-processing of the data to handle the sparsity
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
            np.random.seed(10)
            subset_N = np.random.choice(np.arange(self.N), size=batch_size, replace=False)
            Subs = list(zip(*subs_nz))
            SubsX = list(zip(*subs_X_nz))
        else:
            if self.N > 5000:
                batch_size = 5000
                np.random.seed(10)
                subset_N = np.random.choice(np.arange(self.N), size=batch_size, replace=False)
                Subs = list(zip(*subs_nz))
                SubsX = list(zip(*subs_X_nz))
            else:
                subset_N = None
                Subs = None
                SubsX = None
        print(f'batch_size: {batch_size}\n')

        for r in range(self.num_realizations):

            self._initialize(rng=np.random.RandomState(self.rseed), nodes=nodes)

            self._update_old_variables()
            self._update_cache(data, subs_nz, data_X, subs_X_nz) # type: ignore

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            if flag_conv == 'log':
                loglik = self.inf

            if self.verbose:
                print('Updating realization {0} ...'.format(r), end=' ')
            loglik_values = []
            time_start = time.time()
            # --- single step iteration update ---
            while np.logical_and(not convergence, it < self.max_iter):
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w, delta_beta = self._update_em(data, data_X, subs_nz,
                                                                        subs_X_nz) # type: ignore
                if flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(
                        data,
                        data_X,
                        it,
                        loglik,
                        coincide,
                        convergence,
                        batch_size,
                        subset_N, # type: ignore
                        Subs, # type: ignore
                        SubsX # type: ignore
                    )
                    loglik_values.append(loglik)
                elif flag_conv == 'deltas':
                    it, coincide, convergence = self._check_for_convergence_delta(it, coincide,
                                                                                  delta_u, delta_v,
                                                                                  delta_w,
                                                                                  delta_beta,
                                                                                  convergence)
                else:
                    print(colored('Error! flag_conv can be either "log" or "deltas"', 'red'))
                    break

            if flag_conv == 'log':
                if maxL < loglik:
                    self._update_optimal_parameters()
                    best_loglik = list(loglik_values)
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    # best_r = r
            elif flag_conv == 'deltas':
                if not batch_size:
                    loglik = self.__Likelihood(data, data_X)
                else:
                    loglik = self.__Likelihood_batch(data, data_X, subset_N, Subs, SubsX) # type: ignore
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    # best_r = r
            if self.verbose:
                print(f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds')
                # print(f'Best real = {best_r} - maxL = {maxL} - best iterations = {final_it}')

            self.rseed += self.rng.randint(10000)
            # end cycle over realizations

        if np.logical_and(final_it == self.max_iter, not conv):
            # convergence is not reached
            print('Solution failed to converge in {0} EM steps!'.format(self.max_iter))

        if np.logical_and(self.plot_loglik, flag_conv == 'log'):
            plot_L(best_loglik, int_ticks=True)

        if self.out_inference:
            self.output_results(maxL, nodes, final_it)

        return self.u_f, self.v_f, self.w_f, self.beta_f, maxL

    def _initialize(self, rng: RandomState, nodes: List[int]) -> None:
        """
            Random initialization of the parameters U, V, W, beta.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
            nodes : list
                    List of nodes IDs.
        """

        if self.initialization == 0:
            if self.verbose:
                print('U, V, W and beta are initialized randomly.')
            self._randomize_u_v(rng)
            if self.gamma != 0:
                self._randomize_beta(rng)
            if self.gamma != 1:
                self._randomize_w(rng)

        elif self.initialization == 1:
            if self.verbose:
                print(f'U, V, W and beta are initialized using the input file: {self.files}')
            self._initialize_u(rng, nodes)
            self._initialize_v(rng, nodes)
            self._initialize_beta(rng)
            self._initialize_w(rng)

    def _randomize_u_v(self, rng: RandomState) -> None:
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices U and V, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.u = rng.random_sample((self.N, self.K))
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        if not self.undirected:
            self.v = rng.random_sample((self.N, self.K))
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _randomize_beta(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the beta matrix, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.beta = rng.random_sample((self.K, self.Z))
        self.beta = (self.beta.T / np.sum(self.beta, axis=1)).T

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor W.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)

    def _initialize_u(self, rng: RandomState, nodes: List[int]) -> None:
        """
        Initialize out-going membership matrix u from file.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        nodes : list
                List of nodes IDs.
        """

        self.u = self.theta['u']
        assert np.array_equal(nodes, self.theta['nodes'])

        max_entry = np.max(self.u)
        self.u += max_entry * self.err * rng.random_sample(self.u.shape)

    def _initialize_v(self, rng: RandomState, nodes: List[int]) -> None:
        """
            Initialize in-coming membership matrix v from file.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
            nodes : list
                    List of nodes IDs.
        """

        if self.undirected:
            self.v = self.u
        else:
            self.v = self.theta['v']
            assert np.array_equal(nodes, self.theta['nodes'])

            max_entry = np.max(self.v)
            self.v += max_entry * self.err * rng.random_sample(self.v.shape)

    def _initialize_beta(self, rng: RandomState) -> None:
        """
            Initialize beta matrix beta from file.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        self.beta = self.theta['beta']

        max_entry = np.max(self.beta)
        self.beta += max_entry * self.err * rng.random_sample(self.beta.shape)

    def _initialize_w(self, rng: RandomState) -> None:
        """
            Initialize affinity tensor w from file.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if self.assortative:
            self.w = np.zeros((self.L, self.K))
            for l in range(self.L):
                self.w[l] = np.diag(self.w[l])[np.newaxis, :].copy()
        else:
            self.w = self.theta['w']

        max_entry = np.max(self.w)
        self.w += max_entry * self.err * rng.random_sample(self.w.shape)

    def _update_old_variables(self) -> None:
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        self.w_old = np.copy(self.w)
        self.beta_old = np.copy(self.beta)

    def _update_cache(self,
                      data: Union[skt.dtensor, skt.sptensor],
                      subs_nz: Tuple[np.ndarray],
                      data_X: np.ndarray,
                      subs_X_nz: Tuple[np.ndarray]) -> None:
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
        self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w)
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

    def _lambda0_nz(self,
                    subs_nz: Tuple[np.ndarray],
                    u: np.ndarray,
                    v: np.ndarray,
                    w: np.ndarray) -> np.ndarray:
        """
        Compute the mean lambda0 (M_ij^alpha) for only non-zero entries (denominator of pijkl).

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        nz_recon_I : ndarray
                     Mean lambda0 (M_ij^alpha) for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I

    def _pi0_nz(self,
                subs_X_nz: Tuple[np.ndarray],
                u: np.ndarray,
                v: np.ndarray,
                beta: np.ndarray) -> np.ndarray:
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
            return np.einsum('Ik,kz->Iz', u[subs_X_nz[0], :], beta)
        else:
            return np.einsum('Ik,kz->Iz', u[subs_X_nz[0], :] + v[subs_X_nz[0], :], beta)

    def _update_em(self,
                   data: Union[skt.dtensor, skt.sptensor],
                   data_X: np.ndarray,
                   subs_nz: Tuple[np.ndarray],
                   subs_X_nz: Tuple[np.ndarray]) -> Tuple[float, float, float, float]:
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

        Returns
        -------
        d_u : float
              Maximum distance between the old and the new membership matrix U.
        d_v : float
              Maximum distance between the old and the new membership matrix V.
        d_beta : float
                 Maximum distance between the old and the new beta matrix.
        d_w : float
              Maximum distance between the old and the new affinity tensor W.
        """

        if self.gamma < 1.:
            if not self.assortative:
                d_w = self._update_W(subs_nz)
            else:
                d_w = self._update_W_assortative(subs_nz)
        else:
            d_w = 0
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        if self.gamma > 0.:
            d_beta = self._update_beta(subs_X_nz)
        else:
            d_beta = 0.
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        d_u = self._update_U(subs_nz, subs_X_nz)
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
        else:
            d_v = self._update_V(subs_nz, subs_X_nz)
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        return d_u, d_v, d_w, d_beta

    def _update_W(self, subs_nz):
        """
        Update affinity tensor.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor W.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV

        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q],
                                                   minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))
        non_zeros = Z > 0

        for a in range(self.L):
            self.w[a, non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz: Tuple[np.ndarray]) -> float:
        """
        Update affinity tensor (assuming assortativity).

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor W.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV

        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))
        non_zeros = Z > 0
        for a in range(self.L):
            self.w[a, non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

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
            XUV = np.einsum('Iz,Ik->kz', self.data_pi_nz, self.u[subs_X_nz[0], :])
        else:
            XUV = np.einsum('Iz,Ik->kz', self.data_pi_nz,
                            self.u[subs_X_nz[0], :] + self.v[subs_X_nz[0], :])
        self.beta *= XUV

        row_sums = self.beta.sum(axis=1)
        self.beta[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.beta < self.err_max  # values are too low
        self.beta[low_values_indices] = 0.  # and set to 0.

        dist_beta = np.amax(abs(self.beta - self.beta_old))
        self.beta_old = np.copy(self.beta)

        return dist_beta

    def _update_U(self,
                  subs_nz: Tuple[np.ndarray],
                  subs_X_nz: Tuple[np.ndarray]) -> float:
        """
        Update out-going membership matrix.

        Parameters
        ----------
        subs_nz : Tuple[np.ndarray]
                  Indices of elements of data that are non-zero.
        subs_X_nz : Tuple[np.ndarray]
                    Indices of elements of data_X that are non-zero.

        Returns
        -------
        dist_u : float
                 Maximum distance between the old and the new membership matrix U.
        """

        self.u = self._update_membership(subs_nz, subs_X_nz, self.u, self.v, self.w, self.beta, 1)

        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self,
                  subs_nz: Tuple[np.ndarray],
                  subs_X_nz: Tuple[np.ndarray]) -> float:
        """
        Update in-coming membership matrix.

        Parameters
        ----------
        subs_nz : Tuple[np.ndarray]
                  Indices of elements of data that are non-zero.
        subs_X_nz : Tuple[np.ndarray]
                    Indices of elements of data_X that are non-zero.

        Returns
        -------
        dist_v : float
                 Maximum distance between the old and the new membership matrix V.
        """

        self.v = self._update_membership(subs_nz, subs_X_nz, self.u, self.v, self.w, self.beta, 2)

        row_sums = self.v.sum(axis=1)
        self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_membership(self,
                           subs_nz: Tuple[np.ndarray],
                           subs_X_nz: Tuple[np.ndarray],
                           u: np.ndarray,
                           v: np.ndarray,
                           w: np.ndarray,
                           beta: np.ndarray,
                           m: int) -> np.ndarray:
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

        uttkrp_Xh = np.einsum('Iz,kz->Ik', self.data_pi_nz, beta)

        if self.undirected:
            uttkrp_Xh *= u[subs_X_nz[0]]
        else:
            uttkrp_Xh *= u[subs_X_nz[0]] + v[subs_X_nz[0]]

        uttkrp_DK *= (1 - self.gamma)
        out = uttkrp_DK.copy()
        out[subs_X_nz[0]] += self.gamma * uttkrp_Xh

        return out

    def _check_for_convergence(self,
                               data: Union[skt.dtensor, skt.sptensor],
                               data_X: np.ndarray,
                               it: int,
                               loglik: float,
                               coincide: int,
                               convergence: bool,
                               batch_size: Union[int, None],
                               subset_N: List[int],
                               Subs: List[Tuple[int, int, int]],
                               SubsX: List[Tuple[int, int]]) -> Tuple[
        int, float, int, bool]:
        """
        Check for convergence by using the log-likelihood values.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_X : ndarray
                 Object representing the one-hot encoding version of the design matrix.
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        batch_size : int/None
                     Size of the subset of nodes to compute the likelihood with.
        subset_N : list/None
                   List with a subset of nodes.
        Subs : list/None
               List with elements (a, i, j) of the non zero entries of data.
        SubsX : list
                List with elements (i, z) of the non zero entries of data_X.

        Returns
        -------
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            if not batch_size:
                loglik = self.__Likelihood(data, data_X)
            else:
                loglik = self.__Likelihood_batch(data, data_X, subset_N, Subs, SubsX)
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def __Likelihood(self,
                     data: Union[skt.dtensor, skt.sptensor],
                     data_X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_X : ndarray
                 Object representing the one-hot encoding version of the design matrix.

        Returns
        -------
        l : float
            Log-likelihood value.
        """

        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)
        lG = -self.lambda0_ija.sum()
        logM = np.log(self.lambda0_nz)
        if isinstance(data, skt.dtensor):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, skt.sptensor):
            Alog = data.vals * logM
        lG += Alog.sum()

        if self.undirected:
            logP = np.log(self.pi0_nz)
        else:
            logP = np.log(0.5 * self.pi0_nz)
        if not scipy.sparse.issparse(data_X):
            ind_logP_nz = (np.arange(len(logP)), data_X.nonzero()[1])
            Xlog = data_X[data_X.nonzero()] * logP[ind_logP_nz]
        else:
            Xlog = data_X.data * logP
        lX = Xlog.sum()

        l = (1. - self.gamma) * lG + self.gamma * lX

        if np.isnan(l):
            print("Likelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def __Likelihood_batch(self,
                           data: Union[skt.dtensor, skt.sptensor],
                           data_X: np.ndarray,
                           subset_N: List[int],
                           Subs: List[Tuple[int, int, int]],
                           SubsX: List[Tuple[int, int]]) -> float:
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
        self.lambda0_ija = self._lambda0_full(self.u[subset_N], self.v[subset_N], self.w)
        assert self.lambda0_ija.shape == (self.L, size, size)
        lG = -self.lambda0_ija.sum()
        logM = np.log(self.lambda0_nz)
        IDXs = [i for i, e in enumerate(Subs) if (e[1] in subset_N) and (e[2] in subset_N)]
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

        l = (1. - self.gamma) * lG + self.gamma * lX

        if np.isnan(l):
            print("Likelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def _lambda0_full(self,
                      u: np.ndarray,
                      v: np.ndarray,
                      w: np.ndarray) -> np.ndarray:
        """
        Compute the mean M_ij^alpha for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        M : ndarray
            Mean M_ij^alpha for all entries.
        """

        if w.ndim == 2:
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)

        return M

    def _check_for_convergence_delta(self,
                                     it: int,
                                     coincide: int,
                                     du: float,
                                     dv: float,
                                     dw: float,
                                     db: float,
                                     convergence: bool) -> Tuple[int, int, bool]:
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

        if du < self.convergence_tol and dv < self.convergence_tol and dw < self.convergence_tol and db < self.convergence_tol:
            coincide += 1
        else:
            coincide = 0

        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.beta_f = np.copy(self.beta)

    def output_results(self,
                       maxL: float,
                       nodes: List[int],
                       final_it: int) -> None:
        """
            Output results.

            Parameters
            ----------
            maxL : float
                   Maximum log-likelihood.
            nodes : list
                    List of nodes IDs.
            final_it : int
                       Total number of iterations.
        """
        # Check if the output folder exists, otherwise create it
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)

        outfile = (Path(self.out_folder) / str('theta' + self.end_file)
                   ).with_suffix('.npz')
        np.savez_compressed(outfile,
                            u=self.u_f,
                            v=self.v_f,
                            w=self.w_f,
                            beta=self.beta_f,
                            max_it=final_it,
                            nodes=nodes, maxL=maxL)
        print(f'\nInferred parameters saved in: {outfile.resolve()}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')
