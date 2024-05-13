"""
Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and a pair interaction value.
"""

from __future__ import print_function

import logging
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import sktensor as skt
from typing_extensions import Unpack

from ..input.preprocessing import preprocess
from ..input.tools import (
    check_symmetric, get_item_array_from_subs, log_and_raise_error, sp_uttkrp,
    sp_uttkrp_assortative, transpose_tensor)
from ..output.evaluate import lambda_full
from ..output.plot import plot_L
from .base import FitParams, ModelClass


class JointCRep(ModelClass):  # pylint: disable=too-many-instance-attributes
    """
    Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
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
                 ):
        super().__init__(
            inf,
            err_max,
            err,
            num_realizations,
            convergence_tol,
            decision,
            max_iter,
            plot_loglik,
            flag_conv)

    def check_fit_params(self,
                         initialization: int,
                         eta0: Union[float, None],
                         undirected: bool,
                         assortative: bool,
                         data: Union[skt.dtensor, skt.sptensor],
                         K: int,
                         **extra_params: Unpack[FitParams]
                         ) -> None:

        message = ('The initialization parameter can be either 0, 1, 2 or 3. It is used as an '
                   'indicator to initialize the membership matrices u and v and the affinity '
                   'matrix w. If it is 0, they will be generated randomly; 1 means only '
                   'the affinity matrix w will be uploaded from file; 2 implies the '
                   'membership matrices u and v will be uploaded from file and 3 all u, '
                   'v and w will be initialized through an input file.')
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
        # Call the check_fit_params method from the parent class
        super()._check_fit_params(
            initialization,
            undirected,
            assortative,
            data,
            K,
            available_extra_params,
            data_X=None,
            gamma=None,
            eta0=eta0,
            beta0=None,
            message=message,
            **extra_params)

        # Parameters for the initialization of the model
        self.normalize_rows = False
        self.use_unit_uniform = False

        if self.initialization == 1:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)
            dfW = self.theta['w']
            self.L = dfW.shape[0]
            self.K = dfW.shape[1]

        if "use_approximation" in extra_params:
            self.use_approximation = extra_params["use_approximation"]
        else:
            self.use_approximation = False

        if self.fix_eta:
            self.eta = self.eta_old = self.eta_f = self.eta0

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
        self.check_fit_params(data=data,
                              K=K,
                              initialization=initialization,
                              eta0=eta0,
                              undirected=undirected,
                              assortative=assortative,
                              **extra_params)
        logging.debug('Fixing random seed to: %s', rseed)
        self.rng = np.random.RandomState(rseed)  # pylint: disable=no-member
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood
        self.nodes = nodes

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
            logging.debug('Random number generator seed: %s', self.rng.get_state()[1][0])
            super()._initialize()
            super()._update_old_variables()
            self._update_cache(data, subs_nz)

            # It sets up local variables for convergence checking. coincide and it are counters, convergence is a
            # boolean flag, and loglik is the initial pseudo log-likelihood.
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            logging.debug('Updating realization %s', r)
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
                    it, loglik, coincide, convergence = super()._check_for_convergence(
                        data,
                        it,
                        loglik,
                        coincide,
                        convergence,
                        use_pseudo_likelihood=False)
                    loglik_values.append(loglik)
                    if not it % 100:
                        logging.debug(
                            'Nreal = %s - Log-likelihood = %s - iterations = %s - time = %s  '
                            'seconds', r, loglik, it, np.round(time.time() - time_start, 2)
                        )
                elif self.flag_conv == 'deltas':
                    it, coincide, convergence = super()._check_for_convergence_delta(
                        it,
                        coincide,
                        delta_u,
                        delta_v,
                        delta_w,
                        delta_eta,
                        convergence)

                    if not it % 100:
                        logging.debug(
                            'Nreal = %s - iterations = %s - time = %s seconds',
                            r, it, np.round(time.time() - time_start, 2)
                        )
                else:
                    log_and_raise_error(ValueError, 'flag_conv can be either log or deltas!')

            # After the while loop, it checks if the current pseudo log-likelihood is the maximum
            # so far. If it is, it updates the optimal parameters (
            # self._update_optimal_parameters()) and sets maxL to the current pseudo log-likelihood.
            if self.flag_conv == 'deltas':
                loglik = self._Likelihood(data)

            if maxL < loglik:
                super()._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
                best_r = r

                if self.flag_conv == 'log':
                    best_loglik = list(loglik_values)
            logging.debug('Nreal = %s - Log-likelihood = %s - iterations = %s - '
                          'time = %s seconds', r, loglik, it,
                          np.round(time.time() - time_start, 2))

            # end cycle over realizations

        logging.debug('Best real = %s - maxL = %s - best iterations = %s', best_r, maxL,
                      self.final_it)

        self.maxL = maxL

        if np.logical_and(self.final_it == self.max_iter, not conv):
            # convergence is not reached
            logging.warning('Solution failed to converge in %s EM steps!', self.max_iter)
            logging.warning('Parameters won\'t be saved!')

        else:
            if self.out_inference:
                super()._output_results()

        if np.logical_and(self.plot_loglik, self.flag_conv == 'log'):
            plot_L(best_loglik, int_ticks=True)

        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _update_cache(
            self,
            data: Union[skt.dtensor, skt.sptensor],
            subs_nz: tuple) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda_aij = lambda_full(self.u, self.v, self.w)  # full matrix lambda

        self.lambda_nz = super()._lambda_nz(subs_nz)  # matrix lambda for non-zero entries
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

    def _update_em(
            self,
            data: Union[skt.dtensor, skt.sptensor],
            subs_nz: tuple) -> tuple:
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

    def _update_W(self, subs_nz: Tuple[np.ndarray]) -> float:
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
            log_and_raise_error(ValueError, 'eta update_approx has zero denominator!')

        self.eta = self.AAtSum / den

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)  # type: ignore

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
        if st <= 0:
            log_and_raise_error(ValueError, 'eta fix point has zero denominator!')
        return self.AAtSum / st

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
        self.eta_old = np.copy(self.eta)  # type: ignore

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

    def _Likelihood(self, data: Union[skt.dtensor, skt.sptensor],
                    data_T: Optional[Union[skt.dtensor, skt.sptensor]] = None,
                    data_T_vals: np.ndarray = None,
                    subs_nz: tuple[np.ndarray] = None,
                    T: int = None,
                    mask: np.ndarray = None) -> float:
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
            log_and_raise_error(ValueError, 'log-likelihood is NaN!')

        return l
