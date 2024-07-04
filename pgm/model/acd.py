"""
Class definition of ACD, the algorithm to perform inference in networks with anomaly.
The latent variables are related to community memberships and anomaly parameters.
"""

from __future__ import print_function

import logging
import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson
import sktensor as skt
from typing_extensions import Unpack

from pgm.input.preprocessing import preprocess
from pgm.input.tools import (
    get_item_array_from_subs, log_and_raise_error, sp_uttkrp, sp_uttkrp_assortative,
    transpose_tensor)
from pgm.model.base import ModelBase, ModelFitParameters, ModelUpdateMixin
from pgm.output.evaluate import lambda_full

EPS = 1e-12

class AnomalyDetection(ModelBase, ModelUpdateMixin):
    def __init__(
        self,
        inf=1e10,
        err_max=1e-8,
        err=0.01,
        num_realizations=1,
        convergence_tol=0.1,
        decision=2,
        max_iter=500,
        plot_loglik=False,
        flag_conv: str = "log",
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
        )
        self.inf = inf  # initial value of the log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.num_realizations = num_realizations  # number of iterations with different random initialization
        self.convergence_tol = convergence_tol  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.plot_loglik = plot_loglik
        self.flag_conv = flag_conv

    def check_fit_params(
        self,
        K: int,
        data: Union[skt.sptensor, skt.dtensor],
        undirected: bool,
        initialization: int,
        assortative: bool,
        constrained: bool,
        ag: float,
        bg: float,
        pibr0: float,
        mupr0: float,
        flag_anomaly: bool,
        fix_pibr: bool,
        fix_mupr: bool,
        **extra_params: Unpack[
            ModelFitParameters
        ],  # TODO: This extra_params is gonna be removed soon, I'm keeping here
        # so the check_fit works, but there's a ticket to remove change the logic for the
        # params.
    ) -> None:
        message = "Message"  # TODO: update message
        available_extra_params = [
            "fix_pibr",
            "fix_mubr",
            "fix_beta",
            "fix_w",
            "fix_communities",
            "files",
            "out_inference",
            "out_folder",
            "end_file",
            "verbose",
        ]
        super()._check_fit_params(
            initialization,
            undirected,
            assortative,
            data,
            K,
            available_extra_params,
            message=message,
            data_X=None,
            eta0=None,
            beta0=None,
            gamma=None,
            **extra_params,
        )

        self.assortative = assortative  # if True, the network is assortative
        self.constrained = constrained  # if True, use the configuration with constraints on the updates
        self.ag = ag  # shape of gamma prior
        self.bg = bg  # rate of gamma prior
        self.pibr = pibr0  # pi: anomaly parameter
        self.mupr = mupr0  # mu: prior
        self.flag_anomaly = flag_anomaly
        self.fix_pibr = fix_pibr
        self.fix_mupr = fix_mupr

        if initialization not in {
            0,
            1,
        }:  # indicator for choosing how to initialize u, v and w
            raise ValueError(
                "The initialization parameter can be either 0 or 1. It is used as an indicator to "
                "initialize the membership matrices u and v and the affinity matrix w. If it is 0, they "
                "will be generated randomly, otherwise they will upload from file."
            )
        if self.pibr is not None:
            if (self.pibr < 0) or (self.pibr > 1):
                raise ValueError("The anomaly parameter pibr0 has to be in [0, 1]!")

        if self.mupr is not None:
            if (self.mupr < 0) or (self.mupr > 1):
                raise ValueError("The prior mupr0 has to be in [0, 1]!")

        if self.fix_pibr == True:
            self.pibr = self.pibr_old = self.pibr_f = pibr0
        if self.fix_mupr == True:
            self.mupr = self.mupr_old = self.mupr_f = mupr0

        if self.flag_anomaly == False:
            self.pibr = self.pibr_old = self.pibr_f = 1.0
            self.mupr = self.mupr_old = self.mupr_f = 0.0
            self.fix_pibr = self.fix_mupr = True

        if self.initialization == 1:
            theta = np.load(self.files, allow_pickle=True)
            self.N, self.K = theta["u"].shape

        self.verbose = extra_params.get("verbose", 0)

        # Parameters for the initialization of the model
        self.use_unit_uniform = True
        self.normalize_rows = True

    def fit(
        self,
        data,
        nodes,
        K,
        undirected,
        initialization,
        assortative,
        constrained,
        ag,
        bg,
        pibr0,
        mupr0,
        flag_anomaly,
        fix_pibr,
        fix_mupr,
        mask=None,
        rseed=10,
        **extra_params,
    ):
        """
        Model  networks by using a probabilistic generative model that assume community parameters and
        anomaly parameters. The inference is performed via EM algorithm.

        Parameters
        ----------
        data : ndarray/sptensor
               Graph adjacency tensor.
        data_T: None/sptensor
                Graph adjacency tensor (transpose).
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
        pibr_f : float
                Bernolie parameter.
        mupr_f : float
                prior .
        maxL : float
               Maximum  log-likelihood.
        final_it : int
                   Total number of iterations.
        """
        self.check_fit_params(
            K,
            data,
            undirected,
            initialization,
            assortative,
            constrained,
            ag,
            bg,
            pibr0,
            mupr0,
            flag_anomaly,
            fix_pibr,
            fix_mupr,
            **extra_params,
        )
        logging.debug("Fixing random seed to: %s", rseed)
        self.rseed = rseed  # random seed
        self.rng = np.random.RandomState(self.rseed)  # pylint: disable=no-member

        # Initialize the fit parameters
        maxL = -self.inf  # initialization of the maximum  log-likelihood
        self.nodes = nodes

        # Preprocess the data for fitting the model
        data, data_T, data_T_vals, subs_nz, subs_nz_mask = (
            self._preprocess_data_for_fit(data, mask)
        )

        self.data = data
        self.data_T = data_T
        self.data_T_vals = data_T_vals
        self.mask = mask
        self.subs_nz = subs_nz
        self.subs_nz_mask = subs_nz_mask

        # Run the Expectation-Maximization (EM) algorithm for a specified number of realizations
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
                self._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
                self.best_r = r
                best_loglik_values = list(loglik_values)
            self.rseed += self.rng.randint(100000000)  # TODO: is this really needed?

            # Log the current realization number, log-likelihood, number of iterations, and elapsed time
            self._log_realization_info(
                r, loglik, maxL, self.final_it, self.time_start, convergence
            )

        # end cycle over realizations

        # Store the maximum pseudo log-likelihood
        self.maxL = maxL

        # Evaluate the results of the fitting process
        self._evaluate_fit_results(self.maxL, conv, best_loglik_values)

        return self.u_f, self.v_f, self.w_f, self.pibr_f, self.mupr_f, maxL

    def _initialize_realization(self) -> Tuple[int, bool, int, float, List[float]]:
        """
        This method initializes the parameters for each realization of the EM algorithm.
        It also sets up local variables for convergence checking.
        """

        # Log the current state of the random number generator
        logging.debug("Random number generator seed: %s", self.rng.get_state()[1][0])

        # Initialize the parameters for the current realization
        self._initialize()  # Remark: this version of this method uses the initialize from this
        # class and not from the super, since it has a different logic.

        # Update the old variables for the current realization
        super()._update_old_variables()

        # Update the cache used in the EM update
        self._update_cache(self.data, self.data_T_vals, self.subs_nz)

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

    def _preprocess_data_for_fit(self, data, mask):
        # if data_T is None:
        logging.debug("Preprocessing the data for fitting the model.")
        logging.debug("Data looks like: %s", data)
        data_T = np.einsum("aij->aji", data)
        data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
        # pre-processing of the data to handle the sparsity
        data = preprocess(data)
        data_T = preprocess(data_T)
        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs
        if mask is not None:
            subs_nz_mask = mask.nonzero()
        else:
            subs_nz_mask = None
        return data, data_T, data_T_vals, subs_nz, subs_nz_mask

    def _initialize(self):
        """
        Random initialization of the parameters u, v, w, beta.
        """
        if self.fix_pibr == False:
            self._randomize_pibr()

        if self.fix_mupr == False:
            self._randomize_mupr()

        if self.initialization == 0:
            # Log a message indicating that u, v and w are being initialized randomly
            logging.debug("%s", "u, v and w are initialized randomly.")

            super()._randomize_w()
            super()._randomize_u_v()

        elif self.initialization == 1:
            # Log a message indicating that u, v and w are being initialized using the input file
            logging.debug(
                "u and v are initialized using the input file: %s", self.files
            )

            self.theta = np.load(self.files, allow_pickle=True)
            super()._initialize_u()
            super()._initialize_v()
            self._randomize_w()  # TODO: Check with Hadiseh why this is not using the
            # input file

    def _randomize_pibr(self):
        """
        Generate a random number in (0, 1.).
        """
        self.pibr = self.rng.random_sample(1)[0]

    def _randomize_mupr(self):
        """
        Generate a random number in (0, 1.).
        """
        self.mupr = self.rng.random_sample(1)[0]

    def _initialize_w(
        self, infile_name
    ):  # TODO: Is this method needed? It seems like it should
        # but it is not used anywhere
        """
        Initialize affinity tensor w from file.

        Parameters
        ----------
        infile_name : str
                      Path of the input file.
        """

        with open(infile_name, "rb") as f:
            dfW = pd.read_csv(f, sep="\s+", header=None)
            if self.assortative:
                self.w = np.diag(dfW)[np.newaxis, :].copy()
            else:
                self.w = dfW.values[np.newaxis, :, :]
        if self.fix_w == False:
            max_entry = np.max(self.w)
            self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

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

        # Copy the specific variables
        for var_name in ["pibr", "mupr"]:
            source_var = getattr(self, f"{var_name}{source_suffix}")
            setattr(self, f"{var_name}{target_suffix}", float(source_var))

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

        self.lambda0_nz = super()._lambda_nz(subs_nz)
        if self.assortative == False:
            self.lambda0_nzT = self._lambda0_nz(
                subs_nz, self.v, self.u, np.einsum("akq->aqk", self.w)
            )
        else:
            self.lambda0_nzT = self._lambda0_nz(subs_nz, self.v, self.u, self.w)
        if self.flag_anomaly == True:
            self.Qij_dense, self.Qij_nz = self._QIJ(data, data_T_vals, subs_nz)
        self.M_nz = self.lambda0_nz
        self.M_nz[self.M_nz == 0] = 1

        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz
            if self.flag_anomaly == True:
                self.data_M_nz_Q = data.vals * (1 - self.Qij_nz) / self.M_nz
            else:
                self.data_M_nz_Q = data.vals / self.M_nz

    def _QIJ(self, data, data_T_vals, subs_nz):
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
        data_T_vals : ndarray
            Array with values of entries A[j, i] given non-zero entry (i, j).

        Returns
        -------
        nz_recon_I : ndarray
                     Mean lambda0_ij for only non-zero entries.
        """
        if isinstance(data, skt.dtensor):
            nz_recon_I = np.power(1 - self.pibr, data[subs_nz])
        elif isinstance(data, skt.sptensor):
            nz_recon_I = (
                self.mupr
                * poisson.pmf(data.vals, self.pibr)
                * poisson.pmf(data_T_vals, self.pibr)
            )
            nz_recon_Id = nz_recon_I + (1 - self.mupr) * poisson.pmf(
                data.vals, self.lambda0_nz
            ) * poisson.pmf(data_T_vals, self.lambda0_nzT)
            non_zeros = nz_recon_Id > 0
            nz_recon_I[non_zeros] /= nz_recon_Id[non_zeros]

        lambda0_ija = lambda_full(self.u, self.v, self.w)
        Q_ij_dense = np.ones(lambda0_ija.shape)
        Q_ij_dense *= self.mupr * np.exp(-self.pibr * 2)
        Q_ij_dense_d = Q_ij_dense + (1 - self.mupr) * np.exp(
            -(lambda0_ija + transpose_tensor(lambda0_ija))
        )
        non_zeros = Q_ij_dense_d > 0
        Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros]
        assert np.allclose(Q_ij_dense[0], Q_ij_dense[0].T, rtol=1e-05, atol=1e-08)
        Q_ij_dense[subs_nz] = nz_recon_I

        Q_ij_dense = np.maximum(
            Q_ij_dense, transpose_tensor(Q_ij_dense)
        )  # make it symmetric
        np.fill_diagonal(Q_ij_dense[0], 0.0)

        assert (Q_ij_dense > 1).sum() == 0
        return Q_ij_dense, Q_ij_dense[subs_nz]

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

        if not self.assortative:
            nz_recon_IQ = np.einsum("Ik,Ikq->Iq", u[subs_nz[1], :], w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum("Ik,Ik->Ik", u[subs_nz[1], :], w[subs_nz[0], :])
        nz_recon_I = np.einsum("Iq,Iq->I", nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I

    def _update_em(
        self,
    ):  # , data, data_T_vals, subs_nz, mask=None, subs_nz_mask=None):
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

        Returns
        -------
        d_u : float
              Maximum distance between the old and the new membership matrix u.
        d_v : float
              Maximum distance between the old and the new membership matrix v.
        d_w : float
              Maximum distance between the old and the new affinity tensor w.
        d_pibr : float
                Maximum distance between the old and the new anoamly parameter pi.
        d_mupr : float
                Maximum distance between the old and the new prior mu.
        """

        if self.fix_communities == False:
            d_u = self._update_U()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
            if self.undirected:
                self.v = self.u
                self.v_old = self.v
                d_v = d_u
            else:
                d_v = self._update_V()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)

        else:
            d_u = d_v = 0.0

        if self.fix_w == False:
            if not self.assortative:
                d_w = self._update_W()
            else:
                d_w = self._update_W_assortative()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_w = 0

        if self.fix_pibr == False:
            d_pibr = self._update_pibr(
                self.data,
                self.data_T_vals,
                self.subs_nz,
                mask=self.mask,
                subs_nz_mask=self.subs_nz_mask,
            )
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)

        else:
            d_pibr = 0.0

        if self.fix_mupr == False:
            d_mupr = self._update_mupr(
                self.data,
                self.data_T_vals,
                self.subs_nz,
                mask=self.mask,
                subs_nz_mask=self.subs_nz_mask,
            )
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_mupr = 0.0

        return d_u, d_v, d_w, d_pibr, d_mupr

    def _update_pibr(self, data, data_T_vals, subs_nz, mask=None, subs_nz_mask=None):
        """
        Update  anomaly parameter pi.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.

        Returns
        -------
        dist_pibr : float
                   Maximum distance between the old and the new anomaly parameter pi.
        """
        if isinstance(data, skt.dtensor):
            Adata = (data[subs_nz] * self.Qij_nz).sum()
        elif isinstance(data, skt.sptensor):
            Adata = (data.vals * self.Qij_nz).sum()
        if mask is None:
            self.pibr = Adata / self.Qij_dense.sum()
        else:
            self.pibr = Adata / self.Qij_dense[subs_nz_mask].sum()

        dist_pibr = abs(self.pibr - self.pibr_old)
        self.pibr_old = np.copy(self.pibr)

        return dist_pibr

    def _update_mupr(self, data, data_T_vals, subs_nz, mask=None, subs_nz_mask=None):
        """
        Update prior eta.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.

        Returns
        -------
        dist_mupr : float
                   Maximum distance between the old and the new rprior mu.
        """
        if mask is None:
            self.mupr = self.Qij_dense.sum() / (self.N * (self.N - 1))
        else:
            self.mupr = self.Qij_dense[subs_nz_mask].sum() / (self.N * (self.N - 1))

        dist_mupr = abs(self.pibr - self.mupr_old)
        self.mupr_old = np.copy(self.mupr)

        return dist_mupr

    def _specific_update_U(self):
        """
        Specific logic for updating U in the DynCRep class.
        """

        self.u = (
            self.ag
            - 1
            + self.u_old
            * self._update_membership(self.subs_nz, self.u, self.v, self.w, 1)
        )

        if not self.constrained:
            if self.flag_anomaly == True:
                if self.mask is None:
                    Du = np.einsum("aij,jq->iq", 1 - self.Qij_dense, self.v)
                else:
                    Du = np.einsum(
                        "aij,jq->iq", self.mask * (1 - self.Qij_dense), self.v
                    )
                if not self.assortative:
                    w_k = np.einsum("akq->kq", self.w)
                    Z_uk = np.einsum("iq,kq->ik", Du, w_k)
                else:
                    w_k = np.einsum("ak->k", self.w)
                    Z_uk = np.einsum("ik,k->ik", Du, w_k)

            else:  # flag_anomaly == False
                Du = np.einsum("jq->q", self.v)
                if not self.assortative:
                    w_k = np.einsum("akq->kq", self.w)
                    Z_uk = np.einsum("q,kq->k", Du, w_k)
                else:
                    w_k = np.einsum("ak->k", self.w)
                    Z_uk = np.einsum("k,k->k", Du, w_k)
            Z_uk += self.bg
            non_zeros = Z_uk > 0.0

            if self.flag_anomaly == True:
                self.u[Z_uk == 0] = 0.0
                self.u[non_zeros] /= Z_uk[non_zeros]
            else:
                self.u[:, Z_uk == 0] = 0.0
                self.u[:, non_zeros] /= Z_uk[non_zeros]

        else:
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _specific_update_V(self):  # , subs_nz, mask=None):
        """
        Specific logic for updating V in the DynCRep class.
        """

        self.v = (
            self.ag
            - 1
            + self.v_old
            * self._update_membership(self.subs_nz, self.u, self.v, self.w, 2)
        )

        if not self.constrained:
            if self.flag_anomaly == True:
                if self.mask is None:
                    Dv = np.einsum("aij,ik->jk", 1 - self.Qij_dense, self.u)
                else:
                    Dv = np.einsum(
                        "aij,ik->jk", self.mask * (1 - self.Qij_dense), self.u
                    )
                if not self.assortative:
                    w_k = np.einsum("aqk->qk", self.w)
                    Z_vk = np.einsum("iq,qk->ik", Dv, w_k)
                else:
                    w_k = np.einsum("ak->k", self.w)
                    Z_vk = np.einsum("ik,k->ik", Dv, w_k)

            else:  # flag_anomaly == False
                Dv = np.einsum("ik->k", self.u)
                if not self.assortative:
                    w_k = np.einsum("aqk->qk", self.w)
                    Z_vk = np.einsum("q,qk->k", Dv, w_k)
                else:
                    w_k = np.einsum("ak->k", self.w)
                    Z_vk = np.einsum("k,k->k", Dv, w_k)

            Z_vk += self.bg
            non_zeros = Z_vk > 0

            if self.flag_anomaly == True:
                self.v[Z_vk == 0] = 0.0
                self.v[non_zeros] /= Z_vk[non_zeros]
            else:
                self.v[:, Z_vk == 0] = 0.0
                self.v[:, non_zeros] /= Z_vk[non_zeros]
        else:
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _specific_update_W(self, subs_nz, mask=None, subs_nz_mask=None):
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
        uttkrp_I = self.data_M_nz_Q[:, np.newaxis, np.newaxis] * UV
        for a, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(
                subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L
            )

        self.w = self.ag - 1 + self.w * uttkrp_DKQ

        if self.flag_anomaly == True:
            if mask is None:
                UQk = np.einsum("aij,ik->ajk", (1 - self.Qij_dense), self.u)
            else:
                UQk = np.einsum("aij,ik->ajk", mask * (1 - self.Qij_dense), self.u)
            Z = np.einsum("ajk,jq->akq", UQk, self.v)
        else:  # flag_anomaly == False
            # Z = np.einsum('ik,jq->kq',self.u,self.v)
            Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))[
                np.newaxis, :, :
            ]
        Z += self.bg

        non_zeros = Z > 0
        self.w[Z == 0] = 0.0
        self.w[non_zeros] /= Z[non_zeros]

    def _specific_update_W_assortative(self):
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
        # Let's make some changes!

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum(
            "Ik,Ik->Ik", self.u[self.subs_nz[1], :], self.v[self.subs_nz[2], :]
        )
        uttkrp_I = self.data_M_nz_Q[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(
                self.subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L
            )

        self.w = self.ag - 1 + self.w * uttkrp_DKQ

        if self.flag_anomaly == True:
            if self.mask is None:
                UQk = np.einsum("aij,ik->jk", (1 - self.Qij_dense), self.u)
                Zk = np.einsum("jk,jk->k", UQk, self.v)
                Zk = Zk[np.newaxis, :]
            else:
                Zk = np.einsum(
                    "aij,ijk->ak",
                    self.mask * (1 - self.Qij_dense),
                    np.einsum("ik,jk->ijk", self.u, self.v),
                )
        else:  # flag_anomaly == False
            Zk = np.einsum("ik,jk->k", self.u, self.v)
            Zk = Zk[np.newaxis, :]
        Zk += self.bg

        non_zeros = Zk > 0
        self.w[Zk == 0] = 0
        self.w[non_zeros] /= Zk[non_zeros]

    def _update_membership_Qd(self, subs_nz, u, v, w, m):
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
            uttkrp_DK = sp_uttkrp((1 - self.Qij), subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative((1 - self.Qij), subs_nz, m, u, v, w)

        return uttkrp_DK

    def _update_membership(self, subs_nz, u, v, w, m):
        """
        (Formerly called 'update_membership_Q')

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
            uttkrp_DK = sp_uttkrp(self.data_M_nz_Q, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q, subs_nz, m, u, v, w)

        return uttkrp_DK

    def compute_likelihood(self):
        return self._ELBO(self.data, self.data_T, self.mask, self.subs_nz_mask)

    def _ELBO(self, data, data_T, mask=None, subs_nz_mask=None):
        """
        Compute the  ELBO of the data.

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
            ELBO value.
        """

        self.lambda0_ija = lambda_full(self.u, self.v, self.w)

        if mask is not None:
            Adense = data.toarray()

        if self.flag_anomaly == False:
            l = (data.vals * np.log(self.lambda0_ija[data.subs] + EPS)).sum()
            if mask is None:
                l -= self.lambda0_ija.sum()
            else:
                l -= self.lambda0_ija[subs_nz_mask].sum()
            return l
        else:
            l = 0.0

            # Term containing Q, pi and A

            l -= self.pibr * self.Qij_dense.sum()

            if self.pibr >= 0:
                if mask is None:
                    l += (
                        np.log(self.pibr + EPS)
                        * (self.Qij_dense[data.subs] * data.vals).sum()
                    )
                else:
                    subs_nz = np.logical_and(mask > 0, Adense > 0)
                    l += (
                        np.log(self.pibr + EPS)
                        * (self.Qij_dense[subs_nz] * (Adense[subs_nz])).sum()
                    )

            # Entropy of Bernoulli in Q

            if mask is None:
                non_zeros = self.Qij_dense > 0
                non_zeros1 = (1 - self.Qij_dense) > 0
            else:
                non_zeros = np.logical_and(mask > 0, self.Qij_dense > 0)
                non_zeros1 = np.logical_and(mask > 0, (1 - self.Qij_dense) > 0)

            l -= (
                self.Qij_dense[non_zeros] * np.log(self.Qij_dense[non_zeros] + EPS)
            ).sum()
            l -= (
                (1 - self.Qij_dense)[non_zeros1]
                * np.log((1 - self.Qij_dense)[non_zeros1] + EPS)
            ).sum()

            # Term containing Q, M and A

            if mask is None:
                l -= ((1 - self.Qij_dense) * self.lambda0_ija).sum()
                l += (
                    ((1 - self.Qij_dense)[data.subs])
                    * data.vals
                    * np.log(self.lambda0_ija[data.subs] + EPS)
                ).sum()

                # Term containing Q and mu

                if 1 - self.mupr >= 0:
                    l += np.log(1 - self.mupr + EPS) * (1 - self.Qij_dense).sum()
                if self.mupr >= 0:
                    l += np.log(self.mupr + EPS) * (self.Qij_dense).sum()
            else:
                l -= (
                    (1 - self.Qij_dense[subs_nz_mask]) * self.lambda0_ija[subs_nz_mask]
                ).sum()
                subs_nz = np.logical_and(mask > 0, Adense > 0)
                l += (
                    ((1 - self.Qij_dense)[subs_nz])
                    * data.vals
                    * np.log(self.lambda0_ija[subs_nz] + EPS)
                ).sum()

                if 1 - self.mupr > 0:
                    l += (
                        np.log(1 - self.mupr + EPS)
                        * (1 - self.Qij_dense)[subs_nz_mask].sum()
                    )
                if self.mupr > 0:
                    l += np.log(self.mupr + EPS) * (self.Qij_dense[subs_nz_mask]).sum()

            if self.ag > 1.0:
                l += (self.ag - 1) * np.log(self.u + EPS).sum()
                l += (self.ag - 1) * np.log(self.v + EPS).sum()
            if self.bg > 0.0:
                l -= self.bg * self.u.sum()
                l -= self.bg * self.v.sum()

            if np.isnan(l):
                log_and_raise_error(ValueError, "ELBO is NaN!")
            else:
                return l

    def _log_realization_info(
        self, r, loglik, maxL, final_it, time_start, convergence
    ) -> None:
        """
        Log the current realization number, log-likelihood, number of iterations, and elapsed time.

        Parameters
        ----------
        r : int
            Current realization number.
        loglik : float
            Current log-likelihood.
        final_it : int
            Current number of iterations.
        time_start : float
            Start time of the realization.
        """
        logging.debug(
            "num. realizations = %s - ELBO = %s - ELBOmax = %s - iterations = %s - time = %s "
            "seconds - "
            "convergence = %s",
            r,
            loglik,
            maxL,
            final_it,
            np.round(time.time() - time_start, 2),
            convergence,
        )

    def _update_optimal_parameters(self):
        """
        Update values of the parameters after convergence.
        """

        self._copy_variables(source_suffix="", target_suffix="_f")

        self.pibr_f = np.copy(self.pibr)
        self.mupr_f = np.copy(self.mupr)
        if self.flag_anomaly == True:
            self.Q_ij_dense_f = np.copy(self.Qij_dense)
        else:
            self.Q_ij_dense_f = np.zeros((1, self.N, self.N))

    # def output_results(self, nodes): #TODO: keep until the tests for the outputs is created in
    #  a future ticket
    #     """
    #     Output results.
    #
    #     Parameters
    #     ----------
    #     nodes : list
    #             List of nodes IDs.
    #     """
    #
    #     outfile = (
    #         self.out_folder
    #         + "theta_inf_"
    #         + str(self.flag_anomaly)
    #         + "_"
    #         + self.end_file
    #     )
    #     np.savez_compressed(
    #         outfile + ".npz",
    #         u=self.u_f,
    #         v=self.v_f,
    #         w=self.w_f,
    #         pibr=self.pibr_f,
    #         mupr=self.mupr_f,
    #         max_it=self.final_it,
    #         Q=self.Q_ij_dense_f,
    #         maxL=self.maxL,
    #         nodes=nodes,
    #     )
    #     print(f'\nInferred parameters saved in: {outfile + ".npz"}')
    #     print('To load: theta=np.load(filename), then e.g. theta["u"]')
