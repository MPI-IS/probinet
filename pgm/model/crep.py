"""
Class definition of CRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and reciprocity value.
"""
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import sktensor as skt
from termcolor import colored

from ..input.preprocessing import preprocess
from ..input.tools import get_item_array_from_subs


# NOTE: user changes the class parameters by hand, no yaml file here
class CRep:

    def __init__(self,
                 inf: float = 1e10,  # initial value of the pseudo log-likelihood, aka, infinity
                 err_max: float = 1e-12,  # minimum value for the parameters
                 err: float = 0.1,  # noise for the initialization
                 num_realizations: int = 5,  # number of iterations with different random initialization
                 convergence_tol: float = 1e-4,  # convergence_tol parameter for convergence
                 decision: int = 10,  # convergence parameter
                 max_iter: int = 1000,  # maximum number of EM steps before aborting
                 verbose: bool = True,  # flag to print details
                 flag_conv: str = 'log',  # flag to choose the convergence criterion
                 ) -> None:
        self.inf = inf
        self.err_max = err_max
        self.err = err
        self.num_realizations = num_realizations
        self.convergence_tol = convergence_tol
        self.decision = decision
        self.max_iter = max_iter
        self.verbose = verbose
        self.flag_conv = flag_conv

    def __check_fit_params(self,
                           initialization,
                           eta0,
                           undirected,
                           assortative,
                           data,
                           K,
                           constrained,
                           **extra_params) -> None:
        if initialization not in {
            0, 1, 2, 3
        }:  # indicator for choosing how to initialize u, v and w
            raise ValueError(
                'The initialization parameter can be either 0, 1, 2 or 3. It is used as an '
                'indicator to initialize the membership matrices u and v and the affinity matrix w.'
                'If it is 0, they will be generated randomly; 1 means only the affinity matrix w '
                'will be uploaded from file; 2 implies the membership matrices u and v will be '
                'uploaded from file and 3 all u, v and w will be initialized through an input file.')
        self.initialization = initialization
        if eta0 is not None:
            if (eta0 < 0) or (eta0 > 1):
                raise ValueError(
                    'The reciprocity coefficient eta0 has to be in [0, 1]!')
        self.eta0 = eta0  # initial value for the reciprocity coefficient
        self.undirected = undirected  # flag to call the undirected network
        self.assortative = assortative  # flag to call the assortative network
        self.constrained = constrained  # if True, use the configuration with constraints on the updates

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
            'end_file'
        ]
        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = "Ignoring unrecognised parameter %s." % extra_param
                print(msg)  # Add the warning
                # self.logger.warn(msg) # TODO: check this out
        if "fix_eta" in extra_params:
            self.fix_eta = extra_params["fix_eta"]

            if self.fix_eta:
                if self.eta0 is None:
                    raise ValueError('If fix_eta=True, provide a value for eta0.')
        else:
            self.fix_eta = False

        if "fix_w" in extra_params:
            self.fix_w = extra_params["fix_w"]
            if self.fix_w:
                if self.initialization not in {1, 3}:
                    raise ValueError(
                        'If fix_w=True, the initialization has to be either 1 or 3.'
                    )
        else:
            self.fix_w = False

        if "fix_communities" in extra_params:
            self.fix_communities = extra_params["fix_communities"]
            if self.fix_communities:
                if self.initialization not in {2, 3}:
                    raise ValueError(
                        'If fix_communities=True, the initialization has to be either 2 or 3.'
                    )
        else:
            self.fix_communities = False

        if "files" in extra_params:
            self.files = extra_params["files"]
        if self.initialization > 0:
            self.theta = np.load(self.files.resolve(), allow_pickle=True)

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

        if self.undirected:
            if not (self.fix_eta and self.eta0 == 0):
                raise ValueError(
                    'If undirected=True, the parameter eta has to be fixed equal to 0.'
                )

    def fit(self,
            data: Union[skt.dtensor, skt.sptensor],
            data_T: skt.sptensor,
            data_T_vals: np.ndarray,
            nodes: List[Any],
            rseed: int = 0,
            K: int = 3,
            mask: Optional[np.ndarray] = None,
            initialization: int = 0,
            eta0: Optional[float] = None,
            undirected: bool = False,
            assortative: bool = True,
            constrained: bool = True,
            **extra_params
            ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Model directed networks by using a probabilistic generative model that assume community
        parameters and reciprocity coefficient. The inference is performed via EM algorithm.

        Parameters
        ----------
        data : ndarray/sptensor
               Graph adjacency tensor.
        data_T: None/sptensor
                Graph adjacency tensor (transpose) - if sptensor.
        data_T_vals : None/ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j) - if
                       ndarray.
        nodes : list
                List of nodes IDs.
        flag_conv : str
                    If 'log' the convergence is based on the log-likelihood values; if 'deltas'
                     convergence is based on the differences in the parameters values. The
                     latter is suggested when the dataset is big (N > 1000 ca.).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of
               cross-validation.

        Returns
        -------
        u_f : ndarray
              Out-going membership matrix.
        v_f : ndarray
              In-coming membership matrix.
        w_f : ndarray
              Affinity tensor.
        eta_f : float
                Reciprocity coefficient.
        maxL : float
               Maximum pseudo log-likelihood.
        final_it : int
                   Total number of iterations.
        """
        self.__check_fit_params(data=data,
                                K=K,
                                initialization=initialization,
                                eta0=eta0,
                                undirected=undirected,
                                assortative=assortative,
                                constrained=constrained,
                                **extra_params)
        self.rng = np.random.RandomState(rseed)  # random seed
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood

        if data_T is None:
            E = np.sum(
                data
            )  # weighted sum of edges (needed in the denominator of eta)
            data_T = np.einsum('aij->aji', data)
            data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
            # pre-processing of the data to handle the sparsity
            data = preprocess(data)
            data_T = preprocess(data_T)
        else:
            E = np.sum(data.vals)

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        for r in range(self.num_realizations):

            self._initialize(nodes=nodes)

            self._update_old_variables()
            self._update_cache(data, data_T_vals, subs_nz)

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            if self.verbose:
                print(f'Updating realization {r} ...')
            time_start = time.time()
            # --- single step iteration update ---
            while np.logical_and(not convergence, it < self.max_iter):
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w, delta_eta = self._update_em(
                    data, data_T_vals, subs_nz, denominator=E)
                if self.flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(
                        data,
                        it,
                        loglik,
                        coincide,
                        convergence,
                        data_T=data_T,
                        mask=mask)
                    if self.verbose:
                        if not it % 100:
                            print(
                                f'Nreal = {r} - Pseudo Log-likelihood = {loglik} '
                                f'- iterations = {it} - '
                                f'time = {np.round(time.time() - time_start, 2)} seconds')
                elif self.flag_conv == 'deltas':
                    it, coincide, convergence = self._check_for_convergence_delta(
                        it, coincide, delta_u, delta_v, delta_w, delta_eta,
                        convergence)
                    if self.verbose:
                        if not it % 100:
                            print(
                                f'Nreal = {r} - iterations = {it} - '
                                f'time = {np.round(time.time() - time_start, 2)} seconds'
                            )
                else:
                    raise ValueError('flag_conv can be either log or deltas!')

            if self.flag_conv == 'log':
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    self.final_it = it
                    conv = convergence
            elif self.flag_conv == 'deltas':
                loglik = self._PSLikelihood(data, data_T=data_T, mask=mask)
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    self.final_it = it
                    conv = convergence
            if self.verbose:
                print(
                    f'Nreal = {r} - Pseudo Log-likelihood = {loglik} - iterations = {it} - '
                    f'time = {np.round(time.time() - time_start, 2)} seconds\n'
                )

            # end cycle over realizations

        self.maxPSL = maxL

        if np.logical_and(self.final_it == self.max_iter, not conv):
            # convergence not reached
            try:
                print(
                    colored(
                        'Solution failed to converge in {0} EM steps!'.format(
                            self.max_iter), 'blue'))
            except BaseException:
                print('Solution failed to converge in {0} EM steps!'.format(
                    self.max_iter))

        if self.out_inference:
            self.output_results(nodes)

        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _initialize(self, nodes: List[Any]) -> None:
        """
        Random initialization of the parameters u, v, w, eta.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        nodes : list
                List of nodes IDs.
        """

        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            if self.verbose:
                print('eta is initialized randomly.')
            self._randomize_eta()

        if self.initialization == 0:
            if self.verbose:
                print('u, v and w are initialized randomly.')
            self._randomize_w()
            self._randomize_u_v()

        elif self.initialization == 1:
            if self.verbose:
                print(f'w is initialized using the input file: {self.files}.')
                print('u and v are initialized randomly.')
            self._initialize_w()
            self._randomize_u_v()

        elif self.initialization == 2:
            if self.verbose:
                print(
                    f'u and v are initialized using the input file: {self.files}.'
                )
                print('w is initialized randomly.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._randomize_w()

        elif self.initialization == 3:
            if self.verbose:
                print(
                    f'u, v and w are initialized using the input file: {self.files}.'
                )
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._initialize_w()

    def _randomize_eta(self) -> None:
        """
        Generate a random number in (0, 1.).

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """

        self.eta = self.rng.random_sample(1)[0]

    def _randomize_w(self) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the affinity tensor w.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """
        if self.assortative:
            self.w = self.rng.random_sample((self.L, self.K))
        else:
            self.w = np.zeros((self.L, self.K, self.K))

    def _randomize_u_v(self) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and
        normalize each row.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """

        self.u = self.rng.random_sample((self.N, self.K))
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            self.v = self.rng.random_sample((self.N, self.K))
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _initialize_u(self, nodes: List[Any]) -> None:
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
        self.u += max_entry * self.err * self.rng.random_sample(self.u.shape)

    def _initialize_v(self, nodes: List[Any]) -> None:
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
            self.v += max_entry * self.err * self.rng.random_sample(self.v.shape)

    def _initialize_w(self) -> None:
        """
        Initialize affinity tensor w from file.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
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
        self.eta_old = np.copy(self.eta)

    def _update_cache(self, data: Union[skt.dtensor, skt.sptensor],
                      data_T_vals: np.ndarray, subs_nz: Tuple[np.ndarray]) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda0_nz = self._lambda0_nz(subs_nz)
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals
        self.M_nz[self.M_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz
        self.data_M_nz[self.M_nz == 0] = 0

    def _lambda0_nz(self, subs_nz: Tuple[np.ndarray]) -> np.ndarray:
        """
        Compute the mean lambda0_ij for only non-zero entries.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        nz_recon_I : ndarray
                     Mean lambda0_ij for only non-zero entries.
        """

        if not self.assortative:
            if len(subs_nz) >= 2:
                nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :],
                                        self.w[subs_nz[0], :, :])
            else:
                raise ValueError("subs_nz should have at least 2 elements.")
        else:
            if len(subs_nz) >= 2:
                nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :],
                                        self.w[subs_nz[0], :])
            else:
                raise ValueError("subs_nz should have at least 2 elements.")

        if len(subs_nz) >= 3:
            nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, self.v[subs_nz[2], :])
        else:
            raise ValueError("subs_nz should have at least 3 elements.")

        return nz_recon_I

    def _update_em(self, data: Union[skt.dtensor, skt.sptensor],
                   data_T_vals: np.ndarray, subs_nz: Tuple[np.ndarray],
                   denominator: float) -> Tuple[float, float, float, float]:
        """
        Update parameters via EM procedure.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        denominator : float
                      Denominator used in the update of the eta parameter.

        Returns
        -------
        d_u : float
              Maximum distance between the old and the new membership matrix u.
        d_v : float
              Maximum distance between the old and the new membership matrix v.
        d_w : float
              Maximum distance between the old and the new affinity tensor w.
        d_eta : float
                Maximum distance between the old and the new reciprocity coefficient eta.
        """

        if not self.fix_eta:
            d_eta = self._update_eta(data,
                                     data_T_vals,
                                     denominator=denominator)
        else:
            d_eta = 0.
        self._update_cache(data, data_T_vals, subs_nz)

        if not self.fix_communities:
            d_u = self._update_U(subs_nz)
            self._update_cache(data, data_T_vals, subs_nz)
        else:
            d_u = 0.

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
            self._update_cache(data, data_T_vals, subs_nz)
        else:
            if not self.fix_communities:
                d_v = self._update_V(subs_nz)
                self._update_cache(data, data_T_vals, subs_nz)
            else:
                d_v = 0.

        if not self.fix_w:
            if not self.assortative:
                d_w = self._update_W(subs_nz)
            else:
                d_w = self._update_W_assortative(subs_nz)
            self._update_cache(data, data_T_vals, subs_nz)
        else:
            d_w = 0

        return d_u, d_v, d_w, d_eta

    def _update_eta(self, data: Union[skt.dtensor, skt.sptensor],
                    data_T_vals: np.ndarray,
                    denominator: Optional[float] = None) -> float:
        """
        Update reciprocity coefficient eta.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        denominator : float
                      Denominator used in the update of the eta parameter.

        Returns
        -------
        dist_eta : float
                   Maximum distance between the old and the new reciprocity coefficient eta.
        """

        if denominator is None:
            Deta = data.sum()
        else:
            Deta = denominator

        self.eta *= (self.data_M_nz * data_T_vals).sum() / Deta

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)  # type: ignore

        return dist_eta

    def _update_U(self, subs_nz: Tuple[np.ndarray]) -> float:
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

        self.u = self.u_old * self._update_membership(subs_nz, 1)

        if not self.constrained:
            Du = np.einsum('iq->q', self.v)
            if not self.assortative:
                w_k = np.einsum('akq->kq', self.w)
                Z_uk = np.einsum('q,kq->k', Du, w_k)
            else:
                w_k = np.einsum('ak->k', self.w)
                Z_uk = np.einsum('k,k->k', Du, w_k)
            non_zeros = Z_uk > 0.
            self.u[:, Z_uk == 0] = 0.
            self.u[:, non_zeros] /= Z_uk[np.newaxis, non_zeros]
        else:
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self, subs_nz: Tuple[np.ndarray]) -> float:
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

        if not self.constrained:
            Dv = np.einsum('iq->q', self.u)
            if not self.assortative:
                w_k = np.einsum('aqk->qk', self.w)
                Z_vk = np.einsum('q,qk->k', Dv, w_k)
            else:
                w_k = np.einsum('ak->k', self.w)
                Z_vk = np.einsum('k,k->k', Dv, w_k)
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.
            self.v[:, non_zeros] /= Z_vk[np.newaxis, non_zeros]
        else:
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

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
        if len(subs_nz) < 3:
            raise ValueError("subs_nz should have at least 3 elements.")

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :],
                       self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0],
                                                   weights=uttkrp_I[:, k, q],
                                                   minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = np.einsum('k,q->kq', self.u.sum(axis=0),
                      self.v.sum(axis=0))[np.newaxis, :, :]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

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
                 Maximum distance between the old and the new affinity tensor w.
        """
        if len(subs_nz) < 3:
            raise ValueError("subs_nz should have at least 3 elements.")

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :],
                       self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0],
                                            weights=uttkrp_I[:, k],
                                            minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) *
             (self.v_old.sum(axis=0)))[np.newaxis, :]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_membership(self, subs_nz: Tuple[np.ndarray],
                           m: int) -> np.ndarray:
        """
        Return the Khatri-Rao product (sparse version) used in the update of the membership
        matrices.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
             tensor: if 1 it
            works with the matrix u; if 2 it works with v.

        Returns
        -------
        uttkrp_DK : ndarray
                    Matrix which is the result of the matrix product of the unfolding of the
                    tensor and the
                    Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, self.u, self.v,
                                  self.w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m,
                                              self.u, self.v, self.w)

        return uttkrp_DK

    def _check_for_convergence(self,
                               data: Union[skt.dtensor, skt.sptensor],
                               it: int,
                               loglik: float,
                               coincide: int,
                               convergence: bool,
                               data_T: skt.sptensor,
                               mask: Optional[np.ndarray] = None) -> Tuple[
        int, float, int, bool]:
        """
        Check for convergence by using the pseudo log-likelihood values.

        Parameters
        ----------
        data : sptensor/dtensor
           Graph adjacency tensor.
        it : int
         Number of iteration.
        loglik : float
             Pseudo log-likelihood value.
        coincide : int
               Number of time the update of the pseudo log-likelihood respects the
               convergence_tol.
        convergence : bool
                  Flag for convergence.
        data_T : sptensor/dtensor
             Graph adjacency tensor (transpose).
        mask : ndarray
           Mask for selecting the held out set in the adjacency tensor in case of
           cross-validation.

        Returns
        -------
        it : int
         Number of iteration.
        loglik : float
             Pseudo log-likelihood value.
        coincide : int
               Number of time the update of the pseudo log-likelihood respects the
               convergence_tol.
        convergence : bool
                  Flag for convergence.
        """
        if it % 10 == 0:
            old_L = loglik
            loglik = self._PSLikelihood(data, data_T=data_T, mask=mask)
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def _check_for_convergence_delta(self, it: int, coincide: int, du: float, dv: float,
                                     dw: float, de: float, convergence: bool) -> Tuple[int, int, bool]:
        """
        Check for convergence by using the maximum distances between the old and the new
        parameters values.

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

        if du < self.convergence_tol and dv < self.convergence_tol and dw < self.convergence_tol and \
                de < self.convergence_tol:
            coincide += 1
        else:
            coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _PSLikelihood(self, data: Union[skt.dtensor, skt.sptensor],
                      data_T: skt.sptensor,
                      mask: Optional[np.ndarray] = None) -> float:
        """
        Compute the pseudo log-likelihood of the data.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T : sptensor/dtensor
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of
               cross-validation.

        Returns
        -------
        l : float
            Pseudo log-likelihood value.
        """

        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)

        if mask is not None:
            sub_mask_nz = mask.nonzero()
            if isinstance(data, skt.dtensor):
                l = -self.lambda0_ija[sub_mask_nz].sum(
                ) - self.eta * data_T[sub_mask_nz].sum()
            elif isinstance(data, skt.sptensor):
                l = -self.lambda0_ija[sub_mask_nz].sum(
                ) - self.eta * data_T.toarray()[sub_mask_nz].sum()
        else:
            if isinstance(data, skt.dtensor):
                l = -self.lambda0_ija.sum() - self.eta * data_T.sum()
            elif isinstance(data, skt.sptensor):
                l = -self.lambda0_ija.sum() - self.eta * data_T.vals.sum()
        logM = np.log(self.M_nz)
        if isinstance(data, skt.dtensor):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, skt.sptensor):
            Alog = data.vals * logM

        l += Alog.sum()

        if np.isnan(l):
            print("PSLikelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def _lambda0_full(self, u: np.ndarray, v: np.ndarray,
                      w: np.ndarray) -> np.ndarray:
        """
        Compute the mean lambda0 for all entries.

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
            Mean lambda0 for all entries.
        """

        if w.ndim == 2:
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)
        return M

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = np.copy(self.eta)  # type: ignore

    def output_results(self, nodes: List[Any]) -> None:
        """
        Output results.

        Parameters
        ----------
        nodes : list
                List of nodes IDs.
        """

        # Check if the output folder exists, otherwise create it
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        # Check if the output folder is a Path
        if not isinstance(self.out_folder, Path):
            self.out_folder = Path(self.out_folder)

        outfile = (self.out_folder / str('theta' + self.end_file)).with_suffix('.npz')
        np.savez_compressed(outfile,
                            u=self.u_f,
                            v=self.v_f,
                            w=self.w_f,
                            eta=self.eta_f,
                            max_it=self.final_it,
                            maxPSL=self.maxPSL,
                            nodes=nodes)
        print(f'\nInferred parameters saved in: {outfile.resolve()}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals: np.ndarray, subs: Tuple[np.ndarray], m: int, u: np.ndarray,
              v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the Khatri-Rao product (sparse version).

    Parameters
    ----------
    vals : ndarray
           Values of the non-zero entries.
    subs : tuple
           Indices of elements that are non-zero. It is a n-tuple of array-likes and the length
           of tuple n must be
           equal to the dimension of tensor.
    m : int
        Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
        tensor: if 1 it
        works with the matrix u; if 2 it works with v.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.

    Returns
    -------
    out : ndarray
          Matrix which is the result of the matrix product of the unfolding of the tensor and
          the Khatri-Rao product
          of the membership matrix.
    """
    if len(subs) < 3:
        raise ValueError("subs_nz should have at least 3 elements.")

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) *
                    v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating v
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) *
                    u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals: np.ndarray, subs: Tuple[np.ndarray], m: int,
                          u: np.ndarray, v: np.ndarray,
                          w: np.ndarray) -> np.ndarray:
    """
    Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

    Parameters
    ----------
    vals : ndarray
           Values of the non-zero entries.
    subs : tuple
           Indices of elements that are non-zero. It is a n-tuple of array-likes and the length
           of tuple n must be
           equal to the dimension of tensor.
    m : int
        Mode in which the Khatri-Rao product of the membership matrix is multiplied with the
        tensor: if 1 it
        works with the matrix u; if 2 it works with v.
    u : ndarray
        Out-going membership matrix.
    v : ndarray
        In-coming membership matrix.
    w : ndarray
        Affinity tensor.

    Returns
    -------
    out : ndarray
          Matrix which is the result of the matrix product of the unfolding of the tensor and
          the Khatri-Rao product
          of the membership matrix.
    """
    if len(subs) < 3:
        raise ValueError("subs_nz should have at least 3 elements.")

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(
                tmp.dtype)
        elif m == 2:  # we are updating v
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(
                tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out
