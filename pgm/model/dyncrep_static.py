"""
The original static version of the code. I started integrating this one into the package too
until I realized I could actually do this with the dyncrep.py and temporal attribute set to
False. I decided to keep it for a while until I know that I am not missing any of its features
with the new implementation.
"""

from __future__ import print_function

import logging
import sys
import time

import numpy as np
from scipy.optimize import brentq, root
import sktensor as skt

from ..input.preprocessing import preprocess
from ..input.tools import get_item_array_from_subs, sp_uttkrp, sp_uttkrp_assortative
from ..output.evaluate import u_with_lagrange_multiplier
from ..output.plot import plot_L
from .constants import EPS_
from .dyncrep import u_with_lagrange_multiplier


class CRepDyn:
    def __init__(
        self,
        undirected=False,
        initialization=0,
        ag=1.0,
        bg=0.0,
        rseed=0,
        inf=10000000000.0,
        err_max=0.000000000001,
        err=0.01,
        N_real=1,
        tolerance=0.001,
        decision=10,
        max_iter=500,
        out_inference=False,
        in_parameters="../data/input/synthetic/theta_500_3_5.0_6_0.05_0.2_10",
        fix_communities=False,
        fix_w=False,
        plot_loglik=False,
        beta0=0.5,
        flag_data_T=0,
        out_folder="../data/output/",
        end_file=".dat",
        assortative=True,
        eta0=None,
        fix_eta=False,
        fix_beta=False,
        constrained=False,
        verbose=False,
    ):

        self.undirected = undirected  # flag to call the undirected network
        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the pseudo log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = (
            N_real  # number of iterations with different random initialization
        )
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.end_file = end_file  # output file suffix
        self.assortative = assortative  # if True, the network is assortative
        self.fix_eta = fix_eta  # if True, the eta parameter is fixed
        self.fix_beta = fix_beta  # if True, the beta parameter is fixed
        self.fix_communities = fix_communities
        self.fix_w = fix_w
        self.constrained = constrained  # if True, use the configuration with constraints on the updates
        self.verbose = verbose  # flag to print details

        self.ag = ag  # shape of gamma prior
        self.bg = bg  # rate of gamma prior
        self.eta0 = eta0  # initial value for the reciprocity coefficient
        self.beta0 = beta0
        self.plot_loglik = plot_loglik
        self.flag_data_T = flag_data_T  # if 0: previous time step, 1: same time step
        self.in_parameters = in_parameters

        if initialization not in {
            0,
            1,
            2,
            3,
        }:  # indicator for choosing how to initialize u, v and w
            raise ValueError(
                "The initialization parameter can be either 0, 1 or 2. It is used as an indicator to "
                "initialize the membership matrices u and v and the affinity matrix w. If it is 0, they "
                "will be generated randomly, otherwise they will upload from file."
            )
        self.initialization = initialization
        if self.eta0 is not None:
            if (self.eta0 < 0) or (self.eta0 > 1):
                raise ValueError(
                    "The reciprocity coefficient eta0 has to be in [0, 1]!"
                )
        if self.fix_eta:
            if self.eta0 is None:
                self.eta0 = 0.0

        if self.ag < 1:
            self.ag = 1.0
        if self.bg < 0:
            self.bg = 0.0

        if self.fix_eta:
            self.eta = self.eta_old = self.eta_f = self.eta0

        if self.fix_beta:
            self.beta = self.beta_old = self.beta_f = self.beta0

    # @gl.timeit('fit')
    def fit(self, data, T, nodes, mask=None, K=2):
        """
        Model directed networks by using a probabilistic generative model that assume community parameters and
        reciprocity coefficient. The inference is performed via EM algorithm.
        Parameters
        ----------
        data : ndarray/sptensor
               Graph adjacency tensor.
        data_T: None/sptensor
                Graph adjacency tensor (transpose).
        data_T_vals : None/ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        nodes : list
                List of nodes IDs.
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
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
        self.K = K
        self.N = data.shape[-1]
        self.L = 1

        T = max(0, min(T, data.shape[0] - 1))
        self.T = T
        data = data[: T + 1, :, :]

        # Pre-process data

        data_AtAtm1 = np.zeros(data.shape)
        # data_tm1 = np.zeros_like(data)

        data_AtAtm1[0, :, :] = data[
            0, :, :
        ]  # to calculate numerator containing Aij(t)*(1-Aij(t-1))

        if self.flag_data_T == 1:  # same time step
            self.E0 = np.sum(data[0])  # to calculate denominator eta
            self.Etg0 = np.sum(data[1:])  # to calculate denominator eta
        else:  # previous time step
            self.E0 = 0.0  # to calculate denominator eta
            self.Etg0 = np.sum(data[:-1])  # to calculate denominator eta

        self.bAtAtm1 = 0
        self.Atm11At = 0

        data_T = np.einsum(
            "aij->aji", data
        )  # to calculate denominator containing Aji(t)

        if self.flag_data_T == 1:
            data_Tm1 = data_T.copy()
        if self.flag_data_T == 0:
            data_Tm1 = np.zeros_like(data)
            for i in range(T):
                data_Tm1[i + 1, :, :] = data_T[i, :, :]
        self.sum_datatm1 = data_Tm1[1:].sum()  # needed in the update of beta

        if T > 0:
            bAtAtm1_l = 0
            Atm11At_l = 0
            for i in range(T):
                data_AtAtm1[i + 1, :, :] = data[i + 1, :, :] * (1 - data[i, :, :])
                # calculate Aij(t)*Aij(t-1)
                sub_nz_and = np.logical_and(data[i + 1, :, :] > 0, data[i, :, :] > 0)
                bAtAtm1_l += (
                    (data[i + 1, :, :][sub_nz_and] * data[i, :, :][sub_nz_and])
                ).sum()
                # calculate (1-Aij(t))*Aij(t-1)
                sub_nz_and = np.logical_and(
                    data[i, :, :] > 0, (1 - data[i + 1, :, :]) > 0
                )
                Atm11At_l += (
                    ((1 - data[i + 1, :, :])[sub_nz_and] * data[i, :, :][sub_nz_and])
                ).sum()
            self.bAtAtm1 = bAtAtm1_l
            self.Atm11At = Atm11At_l

        self.sum_data_hat = data_AtAtm1[1:].sum()  # needed in the update of beta

        # data_T_vals = get_item_array_from_subs(data_T, data.nonzero()) # to
        # calculate denominator containing Aji(t)
        # to calculate denominator containing Aji(t)
        data_T_vals = get_item_array_from_subs(data_Tm1, data_AtAtm1.nonzero())

        data_AtAtm1 = preprocess(
            data_AtAtm1
        )  # to calculate numerator containing Aij(t)*(1-Aij(t-1))
        data = preprocess(data)
        data_T = preprocess(data_Tm1)

        # save the indexes of the nonzero entries of Aij(t)*(1-Aij(t-1))
        if isinstance(data_AtAtm1, skt.dtensor):
            subs_nzp = data_AtAtm1.nonzero()
        elif isinstance(data_AtAtm1, skt.sptensor):
            subs_nzp = data_AtAtm1.subs

        # save the indexes of the nonzero entries of  Aij(t)
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        self.beta_hat = np.ones(T + 1)
        if T > 0:
            self.beta_hat[1:] = self.beta0

        # INFERENCE

        maxL = -self.inf  # initialization of the maximum log-likelihood
        rng = np.random.RandomState(self.rseed)

        for r in range(self.N_real):

            self._initialize(rng=rng)
            self._update_old_variables()

            # convergence local variables
            coincide, it = 0, 0
            convergence = False

            if self.verbose:
                print(f"Updating realization {r} ...", end=" ")

            maxL = -self.inf
            loglik_values = []
            time_start = time.time()
            loglik = -self.inf

            while not convergence and it < self.max_iter:
                delta_u, delta_v, delta_w, delta_eta, delta_beta = self._update_em(
                    data_AtAtm1, data_T_vals, subs_nzp, denominator=None
                )
                it, loglik, coincide, convergence = self._check_for_convergence(
                    data_AtAtm1,
                    data_T_vals,
                    subs_nzp,
                    T,
                    r,
                    it,
                    loglik,
                    coincide,
                    convergence,
                    data_T=data_Tm1,
                    mask=mask,
                )

                loglik_values.append(loglik)

            if self.verbose:
                print("done!")
                print(
                    f"Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - "
                    f"time = {np.round(time.time() - time_start, 2)} seconds"
                )

            if maxL < loglik:
                self._update_optimal_parameters()
                self.maxL = loglik
                self.final_it = it
                conv = convergence
                best_loglik_values = list(loglik_values)

        # self.rseed += 1

        # end cycle over realizations

        if self.plot_loglik:
            plot_L(best_loglik_values, int_ticks=True)

        if self.out_inference:
            self.output_results(nodes)

        return self.u_f, self.v_f, self.w_f, self.eta_f, self.beta_f, self.maxL

    def _initialize(self, rng=None):
        """
        Random initialization of the parameters u, v, w, beta.
        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            self._randomize_eta(rng)

        if self.beta0 is not None:
            self.beta = self.beta0
        else:
            self._randomize_beta(rng)

        if self.initialization == 0:
            if self.verbose:
                print("u, v and w are initialized randomly.")
            self._randomize_w(rng=rng)
            self._randomize_u_v(rng=rng)

        elif self.initialization == 2:  # Intentionally left as 2
            if self.verbose:
                print(
                    "w is initialized randomly; u, and v are initialized using the input files:"
                )
                print(self.in_parameters + ".npz")
            theta = np.load(self.in_parameters + ".npz", allow_pickle=True)
            self._initialize_u(theta["u"], rng=rng)
            self._initialize_v(theta["v"], rng=rng)
            self.N = self.u.shape[0]
            self._randomize_w(rng=rng)

        elif self.initialization == 1:
            if self.verbose:
                logging.info(
                    "u, v and w are initialized using the input "
                    "files:" + str(self.in_parameters / ".npz")
                )
            theta = np.load(
                self.in_parameters.with_suffix(".npz"), allow_pickle=True
            )  # TODO: is  this a Path or a string in the
            # chosen logic?
            self._initialize_u(theta["u"], rng=rng)
            self._initialize_v(theta["v"], rng=rng)
            self._initialize_w(theta["w"], rng=rng)
            self.N = self.u.shape[0]

        elif self.initialization == 3:
            if self.verbose:
                print(
                    "u, and v are initialized randomly; w is initialized using the input files:"
                )
                print(self.in_parameters + ".npz")
            theta = np.load(self.in_parameters + ".npz", allow_pickle=True)
            self._randomize_u_v(rng=rng)
            self.N = self.u.shape[0]
            self._initialize_w(theta["w"], rng=rng)

    def _initialize_u(self, u0, rng=None):
        if u0.shape[0] != self.N:
            raise ValueError(
                "u.shape is different that the initialized one.", self.N, u0.shape[0]
            )
        self.u = u0.copy()
        max_entry = np.max(u0)
        self.u += max_entry * self.err * rng.random_sample(self.u.shape)

    def _initialize_v(self, v0, rng=None):
        if v0.shape[0] != self.N:
            raise ValueError(
                "v.shape is different that the initialized one.", self.N, v0.shape[0]
            )
        self.v = v0.copy()
        max_entry = np.max(v0)
        self.v += max_entry * self.err * rng.random_sample(self.v.shape)

    def _initialize_w(self, w0, rng=None):

        if self.assortative:
            self.w = np.zeros((1, self.K), dtype=float)
            self.w[:] = (np.diag(w0)).copy()
        else:
            self.w = np.zeros((1, self.K, self.K), dtype=float)
            self.w[:] = w0.copy()
        print(self.w.shape)

        if not self.fix_w:
            max_entry = np.max(self.w)
            self.w += max_entry * self.err * rng.random_sample(self.w.shape)

    def _randomize_eta(self, rng=None):
        """
        Generate a random number in (0, 1.).
        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.eta = rng.random_sample(1)

    def _randomize_beta(self, rng=None):
        """
        Generate a random number in (0, 1.).
        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.beta = rng.random_sample(1)

    def _randomize_w(self, rng):
        """
        Assign a random number in (0, 1.) to each entry of the affinity tensor w.
        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """
        if self.assortative:
            self.w = np.zeros((self.L, self.K))
        else:
            self.w = np.zeros((self.L, self.K, self.K))

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = rng.random_sample(1)[0]
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = (
                                self.err * rng.random_sample(1)[0]
                            )

    def _randomize_u_v(self, rng=None):
        """
        Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.
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

    def _update_old_variables(self):
        """
        Update values of the parameters in the previous iteration.
        """
        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        self.w_old = np.copy(self.w)
        self.eta_old = np.copy(self.eta)
        self.beta_old = np.copy(self.beta)

    # @gl.timeit('cache')
    def _update_cache(self, data, data_T_vals, subs_nz):
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
        self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w)
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals
        self.M_nz[self.M_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
            self.data_rho2 = (
                (data[subs_nz] * self.eta * data_T_vals) / self.M_nz
            ).sum()
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz
            self.data_rho2 = ((data.vals * self.eta * data_T_vals) / self.M_nz).sum()

    # @gl.timeit('lambda0_nz')
    def _lambda0_nz(self, subs_nz, u, v, w):
        """
        Compute the mean lambda0_ij for only non-zero entries.
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
                     Mean lambda0_ij for only non-zero entries.
        """
        # for t, i, j in zip(*sub_w_nz):

        if not self.assortative:
            nz_recon_IQ = np.einsum("Ik,kq->Iq", u[subs_nz[1], :], w[0, :, :])
        else:
            nz_recon_IQ = np.einsum("Ik,k->Ik", u[subs_nz[1], :], w[0, :])
        nz_recon_I = np.einsum("Iq,Iq->I", nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I

    #    @gl.timeit('update_em')
    def _update_em(self, data_AtAtm1, data_T_vals, subs_nzp, denominator=None):
        # data_t,data_AtAtm1_t, data_T_vals_t,subs_nz,subs_nzp
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

        self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)

        # gl.timer.cum('uvw')
        if not self.fix_communities:
            d_u = self._update_U(subs_nzp)
            self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)
            if self.undirected:
                self.v = self.u
                self.v_old = self.v
                d_v = d_u
            else:
                d_v = self._update_V(subs_nzp)
                self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)
        else:
            d_u = 0
            d_v = 0

        if not self.fix_w:
            if not self.assortative:
                d_w = self._update_W(subs_nzp)
            else:
                d_w = self._update_W_assortative(subs_nzp)
            self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)
        else:
            d_w = 0

        if not self.fix_beta:
            if self.T > 0:
                d_beta = self._update_beta()
                self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)
            else:
                d_beta = 0.0
        else:
            d_beta = 0.0

        if not self.fix_eta:
            # denominator = (data_T_vals * self.beta_hat[subs_nzp[0]]).sum()
            denominator = self.E0 + self.Etg0 * self.beta_hat[-1]
            d_eta = self._update_eta(denominator=denominator)
            self._update_cache(data_AtAtm1, data_T_vals, subs_nzp)
        else:
            d_eta = 0.0

        return d_u, d_v, d_w, d_eta, d_beta

    # @gl.timeit_cum('update_eta')
    def _update_eta(self, denominator):
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
        if denominator > 0:
            self.eta = self.data_rho2 / denominator
        else:
            self.eta = 0.0

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)

        return dist_eta

    # @gl.timeit('update_beta')
    def _update_beta(self):

        # beta_test = root(func_beta_static, self.beta_old, args=(self, data_AtAtm1, data_T_vals, subs_nzp, mask))
        # beta_test = root_scalar(func_beta_static,  args=(self, data_AtAtm1, data_T_vals, subs_nzp, mask))
        self.beta = brentq(func_beta_static, a=0.001, b=0.999, args=(self))
        self.beta_hat[1:] = self.beta

        dist_beta = abs(self.beta - self.beta_old)
        self.beta_old = np.copy(self.beta)

        return dist_beta

    # @gl.timeit_cum('update_U')
    def _update_U(self, subs_nz):
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

        self.u = (self.ag - 1) + self.u_old * (
            +self._update_membership(subs_nz, self.u, self.v, self.w, 1)
        )

        if not self.constrained:
            Du = np.einsum("iq->q", self.v)
            if not self.assortative:
                w_k = np.einsum("akq->kq", self.w)
                Z_uk = np.einsum("q,kq->k", Du, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_uk = np.einsum("k,k->k", Du, w_k)
            Z_uk *= 1.0 + self.beta_hat[self.T] * self.T
            non_zeros = Z_uk > 0.0
            self.u[:, Z_uk == 0] = 0.0
            self.u[:, non_zeros] /= self.bg + Z_uk[np.newaxis, non_zeros]
        # gl.timer.cum('update_mem')
        else:
            Du = np.einsum("iq->q", self.v)
            if not self.assortative:
                w_k = np.einsum("akq->kq", self.w)
                Z_uk = np.einsum("q,kq->k", Du, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_uk = np.einsum("k,k->k", Du, w_k)
            Z_uk *= 1.0 + self.beta_hat[self.T] * self.T
            for i in range(self.u.shape[0]):
                if self.u[i].sum() > self.err_max:
                    u_root = root(
                        u_with_lagrange_multiplier,
                        self.u_old[i],
                        args=(self.u[i], Z_uk),
                    )
                    self.u[i] = u_root.x
        # row_sums = self.u.sum(axis=1)
        # self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        # low_values_indices = self.u < self.err_max  # values are too low
        # self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))

        self.u_old = np.copy(self.u)

        return dist_u

    # @gl.timeit_cum('update_V')
    def _update_V(self, subs_nz):
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

        self.v = (self.ag - 1) + self.v_old * self._update_membership(
            subs_nz, self.u, self.v, self.w, 2
        )

        if not self.constrained:
            Dv = np.einsum("iq->q", self.u)
            if not self.assortative:
                w_k = np.einsum("aqk->qk", self.w)
                Z_vk = np.einsum("q,qk->k", Dv, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_vk = np.einsum("k,k->k", Dv, w_k)
            Z_vk *= 1.0 + self.beta_hat[self.T] * self.T
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.0
            self.v[:, non_zeros] /= self.bg + Z_vk[np.newaxis, non_zeros]
        else:
            Dv = np.einsum("iq->q", self.u)
            if not self.assortative:
                w_k = np.einsum("aqk->qk", self.w)
                Z_vk = np.einsum("q,qk->k", Dv, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_vk = np.einsum("k,k->k", Dv, w_k)
            Z_vk *= 1.0 + self.beta_hat[self.T] * self.T

            for i in range(self.v.shape[0]):
                if self.v[i].sum() > self.err_max:
                    v_root = root(
                        u_with_lagrange_multiplier,
                        self.v_old[i],
                        args=(self.v[i], Z_vk),
                    )
                    self.v[i] = v_root.x
        # row_sums = self.v.sum(axis=1)
        # self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        # low_values_indices = self.v < self.err_max  # values are too low
        # self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    # @gl.timeit_cum('update_W')
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
                 Maximum distance between the old and the new affinity tensor w.
        """
        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum("Ik,Iq->Ikq", self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV

        for a, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(
                subs_nz[0], weights=uttkrp_I[:, k, q], minlength=1
            )[0]

        self.w = (self.ag - 1) + self.w * uttkrp_DKQ

        Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))[
            np.newaxis, :, :
        ]
        Z *= 1.0 + self.beta_hat[self.T] * self.T
        Z += self.bg

        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.0  # self.err_max  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w_old)

        return dist_w

    # @gl.timeit_cum('update_W_ass')
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
                 Maximum distance between the old and the new affinity tensor w.
        """

        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum("Ik,Ik->Ik", self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV

        for a, k in zip(*sub_w_nz):
            uttkrp_DKQ[:, k] += np.bincount(
                subs_nz[0], weights=uttkrp_I[:, k], minlength=1
            )[0]

        # for k in range(self.K):
        #     uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=1)

        self.w = (self.ag - 1) + self.w * uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
        Z *= 1.0 + self.beta_hat[self.T] * self.T
        Z += self.bg

        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.0  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_membership(self, subs_nz, u, v, w, m):
        """
        Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.
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
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        Returns
        -------
        uttkrp_DK : ndarray
                    Matrix which is the result of the matrix product of the unfolding of the tensor and the
                    Khatri-Rao product of the membership matrix.
        """
        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(
                self.data_M_nz, subs_nz, m, u, v, w, temporal=False
            )

        return uttkrp_DK

    def _check_for_convergence(
        self,
        data,
        data_T_vals,
        subs_nz,
        T,
        r,
        it,
        loglik,
        coincide,
        convergence,
        data_T=None,
        mask=None,
    ):
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
                   Number of time the update of the pseudo log-likelihood respects the tolerance.
        convergence : bool
                      Flag for convergence.
        data_T : sptensor/dtensor
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        Returns
        -------
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the pseudo log-likelihood respects the tolerance.
        convergence : bool
                      Flag for convergence.
        """
        if it % 10 == 0:
            old_L = loglik
            loglik = self.__Likelihood(data, data_T, data_T_vals, subs_nz, T, mask=mask)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    # @gl.timeit('Likelihood')
    def __Likelihood(self, data, data_T, data_T_vals, subs_nz, T, mask=None, EPS=EPS_):
        """
        Compute the pseudo log-likelihood of the data.
        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T : sptensor/dtensor
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        Returns
        -------
        l : float
            Pseudo log-likelihood value.
        """
        # self._update_cache(data, data_T_vals, subs_nz)

        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)

        if self.T == 0:
            l = -self.lambda0_ija.sum()

            logM = np.log(self.M_nz)
            if isinstance(data, skt.dtensor):
                Alog = (data[data.nonzero()] * logM).sum()
            elif isinstance(data, skt.sptensor):
                Alog = (data.vals * logM).sum()
            l += Alog

            return l

        else:
            if mask is not None:
                sub_mask_nz = mask.nonzero()
                if isinstance(data, skt.dtensor):
                    # to do
                    l = (
                        -(1 + self.beta0 * T) * self.lambda0_ija[sub_mask_nz].sum()
                        - self.eta
                        * (data_T[sub_mask_nz] * self.beta_hat[sub_mask_nz[0]]).sum()
                    )
                elif isinstance(data, skt.sptensor):
                    l = (
                        -(1 + self.beta0 * T) * self.lambda0_ija[sub_mask_nz].sum()
                        - self.eta
                        * (
                            data_T.toarray()[sub_mask_nz]
                            * self.beta_hat[sub_mask_nz[0]]
                        ).sum()
                    )
            else:
                if isinstance(data, skt.dtensor):
                    # to do
                    l = -(
                        1 + self.beta_hat[self.T] * T
                    ) * self.lambda0_ija.sum() - self.eta * (
                        data_T[0].sum() + self.beta_hat[-1] * data_T[1:].sum()
                    )
                elif isinstance(data, skt.sptensor):
                    l = (
                        -(1 + self.beta_hat[self.T] * T) * self.lambda0_ija.sum()
                        - self.eta * (data_T.sum(axis=(1, 2)) * self.beta_hat).sum()
                    )

            logM = np.log(self.M_nz)
            if isinstance(data, skt.dtensor):
                Alog = (data[data.nonzero()] * logM).sum()
            elif isinstance(data, skt.sptensor):
                Alog = (data.vals * logM).sum()
            l += Alog

            if isinstance(data, skt.dtensor):
                l += (
                    np.log(self.beta_hat[subs_nz[0]] + EPS) * data[data.nonzero()]
                ).sum()
            elif isinstance(data, skt.sptensor):
                l += (np.log(self.beta_hat[subs_nz[0]] + EPS) * data.vals).sum()
            if self.T > 0:
                l += (np.log(1 - self.beta_hat[-1] + EPS) * self.bAtAtm1).sum()
                l += (np.log(self.beta_hat[-1] + EPS) * self.Atm11At).sum()

            if self.ag >= 1.0:
                l += (self.ag - 1) * np.log(self.u + EPS).sum()
                l += (self.ag - 1) * np.log(self.v + EPS).sum()
            if self.bg >= 0.0:
                l -= self.bg * self.u.sum()
                l -= self.bg * self.v.sum()

            if np.isnan(l):
                print("Likelihood is NaN!!!!")
                sys.exit(1)
            else:
                return l

    def _lambda0_full(self, u, v, w):
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
            M = np.einsum("ik,jk->ijk", u, v)
            M = np.einsum("ijk,ak->aij", M, w)
        else:
            M = np.einsum("ik,jq->ijkq", u, v)
            M = np.einsum("ijkq,akq->aij", M, w)
        return M

    def _update_optimal_parameters(self):
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = np.copy(self.eta)
        if not self.fix_beta:
            self.beta_f = np.copy(self.beta_hat[self.T])

    def output_results(self, nodes):
        """
        Output results.
        Parameters
        ----------
        nodes : list
                List of nodes IDs.
        """

        outfile = self.out_folder + "theta" + self.end_file
        np.savez_compressed(
            outfile + ".npz",
            u=self.u_f,
            v=self.v_f,
            w=self.w_f,
            eta=self.eta_f,
            beta=self.beta_f,
            max_it=self.final_it,
            maxL=self.maxL,
            nodes=nodes,
        )
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


# def u_with_lagrange_multiplier(u, x, y):
#     denominator = x.sum() - (y * u).sum()
#     f_ui = x / (y + denominator)
#     # if np.allclose(u.sum() ,0): return 0.
#     if (u < 0).sum() > 0:
#         return 100. * np.ones(u.shape)
#     # return (f_u - u).flatten()
#     return (f_ui - u)


def func_beta_static(beta_t, obj):
    assert isinstance(obj, CRepDyn)
    if obj.assortative:
        lambda0_ija = np.einsum("k,k->k", obj.u.sum(axis=0), obj.w[1:].sum(axis=0))
    else:
        lambda0_ija = np.einsum("k,kq->q", obj.u.sum(axis=0), obj.w[1:].sum(axis=0))
    lambda0_ija = np.einsum("k,k->", obj.v.sum(axis=0), lambda0_ija)

    # lambda0_ija = np.einsum('k,kq->q',obj.u.sum(axis=0), obj.w.sum(axis=0))
    # lambda0_ija = np.einsum('q,q->',obj.v.sum(axis=0), lambda0_ija)

    bt = -(obj.T * lambda0_ija + obj.eta * obj.sum_datatm1)
    bt -= obj.bAtAtm1 / (1 - beta_t)  # adding Aij(t-1)*Aij(t)

    bt += obj.sum_data_hat / beta_t  # adding sum A_hat from 1 to T
    bt += obj.Atm11At / beta_t  # adding Aij(t-1)*(1-Aij(t))
    return bt


def calculate_lambda(u, v, w):
    if w.ndim == 2:
        M = np.einsum("ik,jk->ijk", u, v)
        M = np.einsum("ijk,ak->aij", M, w)
    else:
        M = np.einsum("ik,jq->ijkq", u, v)
        M = np.einsum("ijkq,akq->aij", M, w)
    return M


def likelihood_aggr(u, v, w, B, EPS=EPS_):
    lambda0_ija = calculate_lambda(u, v, w)
    l = -lambda0_ija.sum()
    AlogM = (B * np.log(lambda0_ija + EPS)).sum()
    l += AlogM
    return l
