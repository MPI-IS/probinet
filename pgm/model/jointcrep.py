"""
Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and a pair interaction value.
"""

from __future__ import print_function

import logging
from pathlib import Path
import time
from typing import Any, List, Union

import numpy as np
import sktensor as skt
from typing_extensions import Unpack

from ..input.preprocessing import preprocess
from ..input.tools import (
    check_symmetric, get_item_array_from_subs, log_and_raise_error, sp_uttkrp,
    sp_uttkrp_assortative, transpose_tensor)
from ..output.plot import plot_L
from .crep import FitParams

# TODO: remove repeated parts once mixin is implemented

class JointCRep: # pylint: disable=too-many-instance-attributes
    """
    Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
    """

    def __init__(self, # pylint: disable=too-many-arguments
                 inf: float = 1e10,  # initial value of the log-likelihood
                 err_max: float = 1e-12,  # minimum value for the parameters
                 err: float = 0.1,  # noise for the initialization
                 num_realizations: int = 3,
                 # number of iterations with different random initialization
                 convergence_tol: float = 0.0001,  # convergence_tol parameter for convergence
                 decision: int = 10,  # convergence parameter
                 max_iter: int = 500,  # maximum number of EM steps before aborting
                 plot_loglik: bool = False,  # flag to plot the log-likelihood
                 flag_conv: str = 'log'  # flag to choose the convergence criterion
                 ) -> None:
        self.inf = inf
        self.err_max = err_max
        self.err = err
        self.num_realizations = num_realizations
        self.convergence_tol = convergence_tol
        self.decision = decision
        self.max_iter = max_iter
        self.plot_loglik = plot_loglik
        self.flag_conv = flag_conv

    def __check_fit_params(self, # pylint: disable=too-many-arguments
                           initialization: int,
                           eta0: Union[float, None],
                           undirected: bool,
                           assortative: bool,
                           data: Union[skt.dtensor, skt.sptensor],
                           K: int,
                           **extra_params: Unpack[FitParams]
                           ) -> None:

        if initialization not in {0, 1, 2,
                                  3}:  # indicator for choosing how to initialize u, v and w
            message = ('The initialization parameter can be either 0, 1, 2 or 3. It is used as an '
                       'indicator to initialize the membership matrices u and v and the affinity '
                       'matrix w. If it is 0, they will be generated randomly; 1 means only '
                       'the affinity matrix w will be uploaded from file; 2 implies the '
                       'membership matrices u and v will be uploaded from file and 3 all u, '
                       'v and w will be initialized through an input file.')
            error_type = ValueError
            log_and_raise_error(logging, error_type, message)
        self.initialization = initialization

        if (eta0 is not None) and (
                eta0 <= 0.):  # initial value for the pair interaction coefficient
            message = 'If not None, the eta0 parameter has to be greater than 0.!'
            error_type = ValueError
            log_and_raise_error(logging, error_type, message)

        self.eta0 = eta0  # initial value for the reciprocity coefficient
        self.undirected = undirected  # flag to call the undirected network
        self.assortative = assortative  # flag to call the assortative network

        self.N = data.shape[1]
        self.L = data.shape[0]
        self.K = K

        available_extra_params = [
            'fix_eta',
            'fix_w',
            'fix_communities',
            'files',
            'out_inference',
            'out_folder',
            'end_file',
            'use_approximation'
        ]

        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = f'Ignoring extra parameter {extra_param}.'
                logging.warning(msg)  # Add the warning

        if "fix_eta" in extra_params:
            self.fix_eta = extra_params["fix_eta"]

            if self.fix_eta:
                if self.eta0 is None:
                    message = 'If fix_eta=True, provide a value for eta0.'
                    error_type = ValueError
                    log_and_raise_error(logging, error_type, message)
        else:
            self.fix_eta = False

        if "fix_w" in extra_params:
            self.fix_w = extra_params["fix_w"]
            if self.fix_w:
                if self.initialization not in {1, 3}:
                    message = 'If fix_w=True, the initialization has to be either 1 or 3.'
                    error_type = ValueError
                    log_and_raise_error(logging, error_type, message)
        else:
            self.fix_w = False

        if "fix_communities" in extra_params:
            self.fix_communities = extra_params["fix_communities"]
            if self.fix_communities:
                if self.initialization not in {2, 3}:
                    message = 'If fix_communities=True, the initialization has to be either 2 or 3.'
                    error_type = ValueError
                    log_and_raise_error(logging, error_type, message)
        else:
            self.fix_communities = False

        if "files" in extra_params:
            self.files = extra_params["files"]

        if self.initialization > 0:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)
            if self.initialization == 1:
                dfW = self.theta['w']
                self.L = dfW.shape[0]
                self.K = dfW.shape[1]
            elif self.initialization == 2:
                dfU = self.theta['u']
                self.N, self.K = dfU.shape
            else:
                dfW = self.theta['w']
                dfU = self.theta['u']
                self.L = dfW.shape[0]
                self.K = dfW.shape[1]
                self.N = dfU.shape[0]
                assert self.K == dfU.shape[1]

        if "out_inference" in extra_params:
            # TODO: what happens if this is not given?
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

        if "use_approximation" in extra_params:
            self.use_approximation = extra_params["use_approximation"]
        else:
            self.use_approximation = False

        if self.undirected:
            if not (self.fix_eta and self.eta0 == 1):
                message = 'If undirected=True, the parameter eta has to be fixed equal to 1 (s.t. log(eta)=0).'
                error_type = ValueError
                log_and_raise_error(logging, error_type, message)
        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta = 0.  # pair interaction term

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_old = 0.  # pair interaction coefficient

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_f = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_f = 0.  # pair interaction coefficient

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

        if self.fix_eta:# TODO: Check with Martina what the type of this should be
            self.eta = self.eta_old = self.eta_f = self.eta0 # type: ignore

    def fit(self,
            data: Union[skt.dtensor, skt.sptensor],
            data_T: skt.sptensor,
            data_T_vals: np.ndarray,
            nodes: List[Any],
            rseed: int = 0,
            K: int = 3,
            initialization: int = 0,
            eta0: Union[float, None] = None,
            undirected: bool = False,
            assortative: bool = True,
            **extra_params: Unpack[FitParams]
            ) -> tuple[np.ndarray[Any,
                                  np.dtype[np.float64]],
                       np.ndarray[Any,
                                  np.dtype[np.float64]],
                       np.ndarray[Any,
                                  np.dtype[np.float64]],
                       float,
                       float]:
        """
        Model directed networks by using a probabilistic generative model based on a Bivariate
        Bernoulli distribution that assumes community parameters and a pair interaction
        coefficient as latent variables. The inference is performed via EM algorithm.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T : sptensor
                 Graph adjacency tensor (transpose).
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        nodes : list
                List of nodes IDs.
        rseed : int
                Random seed.
        K : int
            Number of communities.
        initialization : int
                         Indicator for choosing how to initialize u, v and w.
                         If 0, they will be generated randomly; 1 means only the affinity matrix
                         w will be uploaded from file; 2 implies the membership matrices u and
                         v will be uploaded from file and 3 all u, v and w will be initialized
                         through an input file.
        eta0 : float
             Initial value for the reciprocity coefficient.
        undirected : bool
                     Flag to call the undirected network.
        assortative : bool
                      Flag to call the assortative network.
        use_approximation : bool
                            Flag to use the approximation in the updates.
        extra_params : dict
                        Dictionary of extra parameters.

        Returns
        -------
        u_f : ndarray
              Out-going membership matrix.
        v_f : ndarray
              In-coming membership matrix.
        w_f : ndarray
              Affinity tensor.
        eta_f : float
                Pair interaction coefficient.
        maxL : float
               Maximum log-likelihood.
        """
        self.__check_fit_params(data=data,
                                K=K,
                                initialization=initialization,
                                eta0=eta0,
                                undirected=undirected,
                                assortative=assortative,
                                **extra_params)

        self.rng = np.random.RandomState(rseed)  # pylint: disable=no-member
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood

        if data_T is None:
            data_T = np.einsum('aij->aji', data)
            data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
            # pre-processing of the data to handle the sparsity
            data = preprocess(data)

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        self.AAtSum = (data.vals * data_T_vals).sum()

        # The following part of the code is responsible for running the Expectation-Maximization (EM) algorithm for a
        # specified number of realizations (self.num_realizations):
        for r in range(self.num_realizations):

            # For each realization (r), it initializes the parameters, updates the old variables
            # and updates the cache.
            self._initialize(nodes=nodes)
            self._update_old_variables()
            self._update_cache(data, subs_nz)

            # It sets up local variables for convergence checking. coincide and it are counters, convergence is a
            # boolean flag, and loglik is the initial pseudo log-likelihood.
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            logging.info(f'Updating realization {r} ...')
            loglik_values = []
            time_start = time.time()
            # It enters a while loop that continues until either convergence is achieved or the maximum number of
            # iterations (self.max_iter) is reached.
            while np.logical_and(not convergence, it < self.max_iter):
                #  it performs the main EM update (self._update_em(data, data_T_vals, subs_nz, denominator=E))
                # which updates the memberships and calculates the maximum difference
                # between new and old parameters.
                delta_u, delta_v, delta_w, delta_eta = self._update_em(data, subs_nz)

                # Depending on the convergence flag (self.flag_conv), it checks for convergence using either the
                # pseudo log-likelihood values (self._check_for_convergence(data, it, loglik, coincide, convergence,
                # data_T=data_T, mask=mask)) or the maximum distances between the old and the new parameters
                # (self._check_for_convergence_delta(it, coincide, delta_u, delta_v, delta_w, delta_eta, convergence)).
                if self.flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(
                        data, it, loglik, coincide, convergence)
                    loglik_values.append(loglik)
                    if not it % 100:
                        logging.info(f'Nreal = {r} - Log-likelihood = {loglik} - iterations ='
                                 f' {it} - '
                              f'time = {np.round(time.time() - time_start, 2)} seconds')
                elif self.flag_conv == 'deltas':
                    it, coincide, convergence = self._check_for_convergence_delta(
                        it, coincide, delta_u, delta_v, delta_w, delta_eta, convergence)
                    if not it % 100:
                        logging.info(f'Nreal = {r} - iterations = {it} - '
                              f'time = {np.round(time.time() - time_start, 2)} seconds')
                else:
                    message = 'flag_conv can be either log or deltas!'
                    error_type = ValueError
                    log_and_raise_error(logging, error_type, message)

            # After the while loop, it checks if the current pseudo log-likelihood is the maximum so far. If it is,
            # it updates the optimal parameters (self._update_optimal_parameters()) and sets maxL to the current
            # pseudo log-likelihood.
            if self.flag_conv == 'log':
                if maxL < loglik:
                    self._update_optimal_parameters()
                    best_loglik = list(loglik_values)
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    best_r = r
            elif self.flag_conv == 'deltas':
                loglik = self._Likelihood(data)
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    best_r = r
            logging.info(f'Nreal = {r} - Log-likelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds\n')

            # end cycle over realizations

        logging.info(f'Best real = {best_r} - maxL = {maxL} - best iterations = {final_it}')

        if np.logical_and(final_it == self.max_iter, not conv):
            # convergence is not reached
            logging.warning(f'Solution failed to converge in {self.max_iter} EM steps!')

        if np.logical_and(self.plot_loglik, self.flag_conv == 'log'):
            plot_L(best_loglik, int_ticks=True)

        if self.out_inference:
            self._output_results(maxL, nodes, final_it)

        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _initialize(self, nodes: List[Any]) -> None:
        """
        Random initialization of the parameters u, v, w, eta.

        Parameters
        ----------
        nodes : list
                List of nodes IDs.
        """

        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            logging.info('eta is initialized randomly.')
            self._randomize_eta()

        if self.initialization == 0:
            logging.info('u, v and w are initialized randomly.')
            self._randomize_w()
            self._randomize_u_v()

        elif self.initialization == 1:
            logging.info(f'w is initialized using the input file: {self.files}.')
            logging.info(f'u and v are initialized randomly.')
            self._initialize_w()
            self._randomize_u_v()

        elif self.initialization == 2:
            logging.info(f'u and v are initialized using the input file: {self.files}.')
            logging.info(f'w is initialized randomly.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._randomize_w()

        elif self.initialization == 3:
            logging.info(f'u, v and w are initialized using the input file: {self.files}.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._initialize_w()

    def _randomize_eta(self) -> None:
        """
        Generate a random number in (1., 50.).
        """

        self.eta = self.rng.uniform(1.01, 49.99)

    def _randomize_w(self) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the affinity tensor w.
        """

        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = self.rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = self.rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * self.rng.random_sample(1)


    def _randomize_u_v(self) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the membership matrices u and v,
        and normalize each row.
        """

        self.u = self.rng.random_sample(self.u.shape)
        # row_sums = self.u.sum(axis=1)
        # self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            self.v = self.rng.random_sample(self.v.shape)
            # row_sums = self.v.sum(axis=1)
            # self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _initialize_u(self, nodes: List[Any]) -> None:
        """
        Initialize out-going membership matrix u from file.

        Parameters
        ----------
        nodes : list
                List of nodes IDs.
        """

        self.u = self.theta['u']
        assert np.array_equal(nodes, self.theta['nodes'])

        max_entry = np.max(self.u)
        self.u += max_entry * self.err * self.rng.random_sample(self.u.shape)

    def _initialize_v(self, nodes: List[Any]) -> None:
        """
        Initialize in-coming membership matrix v from file.

        Parameters
        ----------
        nodes : list
                List of nodes IDs.
        """

        if self.undirected:
            self.v = self.u
        else:
            self.v = self.theta['v']
            assert np.array_equal(nodes, self.theta['nodes'])

            max_entry = np.max(self.v)
            self.v += max_entry * self.err * self.rng.random_sample(self.v.shape)

    def _initialize_w(self) -> None:
        """
        Initialize affinity tensor w from file.
        """

        if self.assortative:
            self.w = self.theta['w']
            assert self.w.shape == (self.L, self.K)
        else:
            self.w = self.theta['w']

        max_entry = np.max(self.w)
        self.w += max_entry * self.err * self.rng.random_sample(self.w.shape)

    def _update_old_variables(self) -> None:
        """
        Update values of the parameters in the previous iteration.
        """

        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        self.w_old = np.copy(self.w)
        self.eta_old = np.copy(self.eta) # type: ignore

    def _update_cache(self, data: Union[skt.dtensor, skt.sptensor], subs_nz: tuple) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda_aij = self._lambda_full()  # full matrix lambda

        self.lambda_nz = self._lambda_nz(subs_nz)  # matrix lambda for non-zero entries
        lambda_zeros = self.lambda_nz == 0
        self.lambda_nz[lambda_zeros] = 1  # still good because with np.log(1)=0
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.lambda_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.lambda_nz
        self.data_M_nz[lambda_zeros] = 0  # to use in the updates

        self.den_updates = 1 + self.eta * self.lambda_aij  # to use in the updates
        if not self.use_approximation:
            self.lambdalambdaT = np.einsum(
                'aij,aji->aij',
                self.lambda_aij,
                self.lambda_aij)  # to use in Z and eta
            self.Z = self._calculate_Z()

    def _lambda_full(self):
        """
        Compute the mean lambda for all entries.

        Returns
        -------
        M : ndarray
            Mean lambda for all entries.
        """

        if self.w.ndim == 2:
            M = np.einsum('ik,jk->ijk', self.u, self.v)
            M = np.einsum('ijk,ak->aij', M, self.w)
        else:
            M = np.einsum('ik,jq->ijkq', self.u, self.v)
            M = np.einsum('ijkq,akq->aij', M, self.w)

        return M

    def _lambda_nz(self, subs_nz: tuple) -> np.ndarray:
        """
        Compute the mean lambda_ij for only non-zero entries.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        nz_recon_I : ndarray
                     Mean lambda_ij for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :], self.w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, self.v[subs_nz[2], :])

        return nz_recon_I

    def _calculate_Z(self) -> np.ndarray:
        """
        Compute the normalization constant of the Bivariate Bernoulli distribution.

        Returns
        -------
        Z : ndarray
            Normalization constant Z of the Bivariate Bernoulli distribution.
        """

        Z = self.lambda_aij + transpose_tensor(self.lambda_aij) + self.eta * self.lambdalambdaT + 1
        for _, z in enumerate(Z):
            assert check_symmetric(z)

        return Z

    def _update_em(self, data: Union[skt.dtensor, skt.sptensor], subs_nz: tuple) -> tuple:
        """
        Update parameters via EM procedure.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        d_u : float
              Maximum distance between the old and the new membership matrix u.
        d_v : float
              Maximum distance between the old and the new membership matrix v.
        d_w : float
              Maximum distance between the old and the new affinity tensor w.
        d_eta : float
                Maximum distance between the old and the new pair interaction coefficient eta.
        """

        if not self.fix_communities:
            if self.use_approximation:
                d_u = self._update_U_approx(subs_nz)
            else:
                d_u = self._update_U(subs_nz)
            self._update_cache(data, subs_nz)
        else:
            d_u = 0.

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
            self._update_cache(data, subs_nz)
        else:
            if not self.fix_communities:
                if self.use_approximation:
                    d_v = self._update_V_approx(subs_nz)
                else:
                    d_v = self._update_V(subs_nz)
                self._update_cache(data, subs_nz)
            else:
                d_v = 0.

        if not self.fix_w:
            if not self.assortative:
                if self.use_approximation:
                    d_w = self._update_W_approx(subs_nz)
                else:
                    d_w = self._update_W(subs_nz)
            else:
                if self.use_approximation:
                    d_w = self._update_W_assortative_approx(subs_nz)
                else:
                    d_w = self._update_W_assortative(subs_nz)
            self._update_cache(data, subs_nz)
        else:
            d_w = 0.

        if not self.fix_eta:
            self.lambdalambdaT = np.einsum(
                'aij,aji->aij',
                self.lambda_aij,
                self.lambda_aij)  # to use in Z and eta
            if self.use_approximation:
                d_eta = self._update_eta_approx()
            else:
                d_eta = self._update_eta()
            self._update_cache(data, subs_nz)
        else:
            d_eta = 0.

        return d_u, d_v, d_w, d_eta

    def _update_U_approx(self, subs_nz: tuple) -> float:
        """
        Update out-going membership matrix.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_u : float
                 Maximum distance between the old and the new membership matrix u.
        """

        self.u *= self._update_membership(subs_nz, 1)

        if not self.assortative:
            VW = np.einsum('jq,akq->ajk', self.v, self.w)
        else:
            VW = np.einsum('jk,ak->ajk', self.v, self.w)
        den = np.einsum('aji,ajk->ik', self.den_updates, VW)

        non_zeros = den > 0.
        self.u[den == 0] = 0.
        self.u[non_zeros] /= den[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_U(self, subs_nz: tuple) -> float:
        """
        Update out-going membership matrix.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_u : float
                 Maximum distance between the old and the new membership matrix u.
        """

        self.u *= self._update_membership(subs_nz, 1)

        if not self.assortative:
            VW = np.einsum('jq,akq->ajk', self.v, self.w)
        else:
            VW = np.einsum('jk,ak->ajk', self.v, self.w)
        VWL = np.einsum('aji,ajk->aijk', self.den_updates, VW)
        den = np.einsum('aijk,aij->ik', VWL, 1. / self.Z)

        non_zeros = den > 0.
        self.u[den == 0] = 0.
        self.u[non_zeros] /= den[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V_approx(self, subs_nz: tuple) -> float:
        """
        Update in-coming membership matrix.
        Same as _update_U but with:
        data <-> data_T
        w <-> w_T
        u <-> v

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_v : float
                 Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(subs_nz, 2)

        if not self.assortative:
            UW = np.einsum('jq,aqk->ajk', self.u, self.w)
        else:
            UW = np.einsum('jk,ak->ajk', self.u, self.w)
        den = np.einsum('aij,ajk->ik', self.den_updates, UW)

        non_zeros = den > 0.
        self.v[den == 0] = 0.
        self.v[non_zeros] /= den[non_zeros]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_V(self, subs_nz: tuple) -> float:
        """
        Update in-coming membership matrix.
        Same as _update_U but with:
        data <-> data_T
        w <-> w_T
        u <-> v

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_v : float
                 Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(subs_nz, 2)

        if not self.assortative:
            UW = np.einsum('jq,aqk->ajk', self.u, self.w)
        else:
            UW = np.einsum('jk,ak->ajk', self.u, self.w)
        UWL = np.einsum('aij,ajk->aijk', self.den_updates, UW)
        den = np.einsum('aijk,aij->ik', UWL, 1. / self.Z)

        non_zeros = den > 0.
        self.v[den == 0] = 0.
        self.v[non_zeros] /= den[non_zeros]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_W_approx(self, subs_nz: tuple) -> float:
        """
        Update affinity tensor.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0],
                                                   weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        den = np.einsum('jq,aijk->akq', self.v, UL)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative_approx(self, subs_nz: tuple) -> float:
        """
        Update affinity tensor (assuming assortativity).

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        den = np.einsum('jk,aijk->ak', self.v, UL)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W(self, subs_nz: tuple) -> float:
        """
        Update affinity tensor.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0],
                                                   weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        num = np.einsum('jq,aijk->aijkq', self.v, UL)
        den = np.einsum('aijkq,aij->akq', num, 1. / self.Z)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz: tuple) -> float:
        """
        Update affinity tensor (assuming assortativity).

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        dist_w : float
                 Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        num = np.einsum('jk,aijk->aijk', self.v, UL)
        den = np.einsum('aijk,aij->ak', num, 1. / self.Z)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_eta_approx(self) -> float:
        """
        Update pair interaction coefficient eta.

        Returns
        -------
        dist_eta : float
                   Maximum distance between the old and the new pair interaction coefficient eta.
        """

        den = self.lambdalambdaT.sum()
        if not den > 0.:
            message = 'eta update_approx has zero denominator!'
            error_type = ValueError
            log_and_raise_error(logging, error_type, message)

        self.eta = self.AAtSum / den

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta) # type: ignore

        return dist_eta

    def eta_fix_point(self) -> float:
        """
        Compute the fix point of the pair interaction coefficient eta.
        Returns
        -------
        eta : float
              Fix point of the pair interaction coefficient eta.
        """
        st = (self.lambdalambdaT / self.Z).sum()
        if st > 0:
            return self.AAtSum / st

        message = 'eta fix point has zero denominator!'
        error_type = ValueError
        log_and_raise_error(logging, error_type, message)

    def _update_eta(self) -> float:
        """
        Update pair interaction coefficient eta.

        Returns
        -------
        dist_eta : float
                   Maximum distance between the old and the new pair interaction coefficient eta.
        """

        self.eta = self.eta_fix_point()

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta) # type: ignore

        return dist_eta

    def _update_membership(self, subs_nz: tuple, m: int) -> np.ndarray:
        """
        Return the Khatri-Rao product (sparse version) used in the update of the membership
        matrices.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
            tensor: if 1 itworks with the matrix u; if 2 it works with v.

        Returns
        -------
        uttkrp_DK : ndarray
                    Matrix which is the result of the matrix product of the unfolding of the tensor
                     and the Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, self.u, self.v, self.w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m, self.u, self.v, self.w)

        return uttkrp_DK

    def _check_for_convergence(self,
                               data: Union[skt.dtensor,
                                           skt.sptensor],
                               it: int,
                               loglik: float,
                               coincide: int,
                               convergence: bool) -> tuple:
        """
        Check for convergence by using the log-likelihood values.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        it : int
             Number of iteration.
        loglik : float
                 Pseudo log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.

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
            loglik = self._Likelihood(data)
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def _check_for_convergence_delta(
            self,
            it: int,
            coincide: int,
            du: float,
            dv: float,
            dw: float,
            de: float,
            convergence: bool) -> tuple:
        """
        Check for convergence by using the maximum distances between the old and the new parameters
        values.

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
        de : float
             Maximum distance between the old and the new eta parameter.
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

        if (du < self.convergence_tol and dv < self.convergence_tol and dw < self.convergence_tol
                and de < self.convergence_tol):
            coincide += 1
        else:
            coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _Likelihood(self, data: Union[skt.dtensor, skt.sptensor]) -> float:
        """
        Compute the log-likelihood of the data.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.

        Returns
        -------
        l : float
            Log-likelihood value.
        """

        self.lambdalambdaT = np.einsum(
            'aij,aji->aij',
            self.lambda_aij,
            self.lambda_aij)  # to use in Z and eta
        self.Z = self._calculate_Z()

        ft = (data.vals * np.log(self.lambda_nz)).sum()

        st = 0.5 * np.log(self.eta) * self.AAtSum

        tt = 0.5 * np.log(self.Z).sum()

        l = ft + st - tt

        if np.isnan(l):
            message = 'log-likelihood is NaN!'
            error_type = ValueError
            log_and_raise_error(logging, error_type, message)

        return l

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = np.copy(self.eta) # type: ignore 

    def _output_results(self, maxL: float, nodes: List[Any], final_it: int) -> None:
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

        # Save the inferred parameters
        outfile = (Path(self.out_folder) / str('theta' + self.end_file)).with_suffix('.npz')
        np.savez_compressed(outfile,
                            u=self.u_f,
                            v=self.v_f,
                            w=self.w_f,
                            eta=self.eta_f,
                            max_it=final_it,
                            maxL=maxL,
                            nodes=nodes)
        logging.info(f'Inferred parameters saved in: {outfile.resolve()}')
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')
