"""
Class definition of MTCov, the generative algorithm that incorporates both the topology of interactions and node
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
from ..input.tools import log_and_raise_error, sp_uttkrp, sp_uttkrp_assortative
from ..output.plot import plot_L
from .base import FitParams, ModelClass


class MTCov(ModelClass):
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
                 plot_loglik: bool = False,  # flag to plot the log-likelihood
                 flag_conv: str = 'log'  # flag to choose the convergence criterion
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
            flag_conv)

    def check_fit_params(self,
                         initialization: int,
                         gamma: float,
                         undirected: bool,
                         assortative: bool,
                         data: Union[skt.dtensor, skt.sptensor, np.ndarray],
                         data_X: Union[skt.dtensor, skt.sptensor, np.ndarray],
                         K: int,
                         **extra_params: Unpack[FitParams]
                         ) -> None:

        message = (
            'The initialization parameter can be either 0 or 1. It is used as an indicator to '
            'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
            'will be generated randomly, otherwise they will upload from file.')
        available_extra_params = [
            'files',
            'out_inference',
            'out_folder',
            'end_file',
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
            gamma=gamma,
            message=message,
            **extra_params)

        # Parameters for the initialization of the model
        self.normalize_rows = False

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

    def fit(self,
            data: Union[skt.dtensor, skt.sptensor],
            data_X: np.ndarray,
            nodes: List[Any],
            flag_conv: str = 'log',
            batch_size: Optional[int] = None,
            gamma: float = 0.5,
            rseed: int = 0,
            K: int = 3,
            initialization: int = 0,
            undirected: bool = False,
            assortative: bool = True,
            **extra_params: Unpack[FitParams]
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
        self.check_fit_params(data=data,
                              data_X=data_X,
                              K=K,
                              initialization=initialization,
                              gamma=gamma,
                              undirected=undirected,
                              assortative=assortative,
                              **extra_params
                              )

        self.rng = np.random.RandomState(rseed)  # pylint: disable=no-member
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood
        self.nodes = nodes

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
        logging.debug('batch_size: %s', batch_size)

        for r in range(self.num_realizations):

            super()._initialize()

            super()._update_old_variables()
            self._update_cache(data, subs_nz, data_X, subs_X_nz)  # type: ignore

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            if flag_conv == 'log':
                loglik = self.inf

            logging.debug('Updating realization %s ...', r)
            loglik_values = []
            time_start = time.time()
            # --- single step iteration update ---
            while np.logical_and(not convergence, it < self.max_iter):
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w, delta_beta = self._update_em(data, data_X, subs_nz,
                                                                        subs_X_nz)  # type: ignore
                if flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(
                        data,
                        data_X,
                        it,
                        loglik,
                        coincide,
                        convergence,
                        batch_size,
                        subset_N,  # type: ignore
                        Subs,  # type: ignore
                        SubsX  # type: ignore
                    )
                    loglik_values.append(loglik)
                elif flag_conv == 'deltas':
                    it, coincide, convergence = super()._check_for_convergence_delta(
                        it, coincide, delta_u, delta_v, delta_w, delta_beta, convergence)
                else:
                    log_and_raise_error(ValueError, 'Error! flag_conv can be either "log" or '
                                                    '"deltas"')

            if flag_conv == 'log':
                if maxL < loglik:
                    super()._update_optimal_parameters()
                    best_loglik = list(loglik_values)
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    # best_r = r
            elif flag_conv == 'deltas':
                if not batch_size:
                    loglik = self.__Likelihood(data, data_X)
                else:
                    loglik = self.__Likelihood_batch(
                        data, data_X, subset_N, Subs, SubsX)  # type: ignore
                if maxL < loglik:
                    super()._update_optimal_parameters()
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    # best_r = r
            logging.debug('Nreal = %s - Loglikelihood = %s - iterations = %s - time = '
                          '%s seconds', r, loglik, it,
                          np.round(time.time() - time_start, 2))

            # end cycle over realizations

        if np.logical_and(final_it == self.max_iter, not conv):
            # convergence is not reached
            logging.warning('Solution failed to converge in %s EM steps!', self.max_iter)

        if np.logical_and(self.plot_loglik, flag_conv == 'log'):
            plot_L(best_loglik, int_ticks=True)
            
        self.final_it = final_it
        self.maxL = maxL

        if self.out_inference:
            super()._output_results()

        return self.u_f, self.v_f, self.w_f, self.beta_f, maxL

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

    def _update_W_assortative(self, subs_nz):
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

        self.lambda0_ija = super()._lambda_full()
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
            raise ValueError("Likelihood is NaN!!!!")

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
        self.lambda0_ija = super()._lambda_full()
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
            logging.error("Likelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def _check_for_convergence_delta(self, it, coincide, du, dv, dw, db, convergence):
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
