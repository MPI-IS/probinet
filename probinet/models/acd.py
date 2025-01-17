"""
Class definition of ACD, the algorithm to perform inference in networks with anomaly.
The latent variables are related to community memberships and anomaly parameters
:cite:`safdari2022anomaly` .
"""

import logging
import time
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sparse import COO

from probinet.evaluation.expectation_computation import (
    compute_mean_lambda0,
    compute_mean_lambda0_nonzero,
)

from ..input.preprocessing import preprocess_adjacency_tensor
from ..types import (
    ArraySequence,
    EndFileType,
    FilesType,
    GraphDataType,
    MaskType,
    SubsNzType,
)
from ..utils.matrix_operations import sp_uttkrp, sp_uttkrp_assortative, transpose_tensor
from ..utils.tools import (
    get_item_array_from_subs,
    get_or_create_rng,
    log_and_raise_error,
)
from .base import ModelBase, ModelUpdateMixin
from .classes import GraphData
from .constants import AG_DEFAULT, BG_DEFAULT, EPS_, K_DEFAULT, OUTPUT_FOLDER


class AnomalyDetection(ModelBase, ModelUpdateMixin):
    """
    Class definition of AnomalyDetection, the algorithm to perform inference and anomaly
    detection on networks with reciprocity.
    """

    def __init__(
        self,
        convergence_tol: float = 1e-1,  # Overriding the base class default
        decision: int = 2,  # Overriding the base class default
        err: float = 1e-2,  # Overriding the base class default
        err_max: float = 1e-8,  # Overriding the base class default
        num_realizations: int = 1,  # Overriding the base class default
        max_iter: int = 500,  # Overriding the base class default
        **kwargs,  # Capture all other arguments for ModelBase
    ) -> None:
        # Pass the overridden arguments along with any others to the parent class
        super().__init__(
            convergence_tol=convergence_tol,
            decision=decision,
            err=err,
            err_max=err_max,
            num_realizations=num_realizations,
            max_iter=max_iter,
            **kwargs,  # Forward any other arguments to the base class
        )

        self.__doc__ = ModelBase.__init__.__doc__

    def _check_fit_params(self, **kwargs) -> None:
        # Call the check_fit_params method from the parent class
        super()._check_fit_params(**kwargs)

        self.constrained = kwargs.get(
            "constrained", False
        )  # if True, use the configuration with constraints on the updates
        self.ag = kwargs.get("ag", 1.5)  # shape of gamma prior
        self.bg = kwargs.get("bg", 10.0)  # rate of gamma prior
        self.flag_anomaly = kwargs.get("flag_anomaly", True)
        self.fix_pibr = kwargs.get("fix_pibr", False)
        self.fix_mupr = kwargs.get("fix_mupr", False)
        self.pibr = kwargs.get("pibr0", None)  # pi: anomaly parameter
        self.mupr = kwargs.get("mupr0", None)  # mu: prior

        if self.pibr is not None:
            if (self.pibr < 0) or (self.pibr > 1):
                raise ValueError("The anomaly parameter pibr0 has to be in [0, 1]!")

        if self.mupr is not None:
            if (self.mupr < 0) or (self.mupr > 1):
                raise ValueError("The prior mupr0 has to be in [0, 1]!")

        if self.fix_pibr == True:
            self.pibr_old = self.pibr_f = self.pibr
        if self.fix_mupr == True:
            self.mupr_old = self.mupr_f = self.mupr

        if not self.flag_anomaly:
            self.pibr = self.pibr_old = self.pibr_f = 1.0
            self.mupr = self.mupr_old = self.mupr_f = 0.0
            self.fix_pibr = self.fix_mupr = True

        if self.initialization == 1:
            theta = np.load(self.files, allow_pickle=True)
            self.N, self.K = theta["u"].shape

        # Parameters for the initialization of the models
        self.use_unit_uniform = True
        self.normalize_rows = True

    def fit(
        self,
        gdata: GraphData,
        ag: float = AG_DEFAULT,
        bg: float = BG_DEFAULT,
        pibr0: Optional[float] = None,
        mupr0: Optional[float] = None,
        flag_anomaly: bool = True,
        fix_pibr: bool = False,
        fix_mupr: bool = False,
        K: int = K_DEFAULT,
        undirected: bool = False,
        initialization: int = 0,
        assortative: bool = True,
        constrained: bool = False,
        fix_w: bool = False,
        fix_communities: bool = False,
        mask: Optional[MaskType] = None,
        out_inference: bool = True,
        out_folder: Path = OUTPUT_FOLDER,
        end_file: Optional[EndFileType] = None,
        files: Optional[FilesType] = None,
        rng: Optional[np.random.Generator] = None,
        **__kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """
        Fit the AnomalyDetection models to the provided data.

        Parameters
        ----------
        gdata
            Graph adjacency tensor.
        ag
            Shape of gamma prior, by default 1.5.
        bg
            Rate of gamma prior, by default 10.0.
        pibr0
            Initial value for the anomaly parameter pi, by default None.
        mupr0
            Initial value for the prior mu parameter, by default None.
        flag_anomaly
            If True, the anomaly detection is enabled, by default True.
        fix_pibr
            If True, the anomaly parameter pi is fixed, by default False.
        fix_mupr
            If True, the prior mu parameter is fixed, by default False.
        K
            Number of communities, by default 3.
        undirected
            If True, the graph is considered undirected, by default False.
        initialization
            Indicator for choosing how to initialize u, v, and w, by default 0.
        assortative
            If True, the network is considered assortative, by default True.
        constrained
            If True, constraints are applied on the updates, by default False.
        fix_w
            If True, the affinity tensor w is fixed, by default False.
        fix_communities
            If True, the community memberships are fixed, by default False.
        mask
            Mask for selecting the held-out set in the adjacency tensor in case of cross-validation, by default None.
        out_inference
            If True, evaluation inference results, by default True.
        out_folder
            Output folder for inference results, by default "outputs/".
        end_file
            Suffix for the evaluation file, by default None.
        files
            Path to the file for initialization, by default None.
        rng
            Random number generator, by default None.
        **kwargs
            Additional parameters for the model.

        Returns
        -------
        u_f
            Final out-going membership matrix.
        v_f
            Final in-coming membership matrix.
        w_f
            Final affinity tensor.
        pibr_f
            Final anomaly parameter pi.
        mupr_f
            Final prior mu parameter.
        maxL
            Maximum log-likelihood.
        """

        # Check the input parameters
        self._check_fit_params(
            K=K,
            data=gdata.adjacency_tensor,
            undirected=undirected,
            initialization=initialization,
            assortative=assortative,
            constrained=constrained,
            ag=ag,
            bg=bg,
            pibr0=pibr0,
            mupr0=mupr0,
            flag_anomaly=flag_anomaly,
            fix_pibr=fix_pibr,
            fix_mupr=fix_mupr,
            fix_communities=fix_communities,
            fix_w=fix_w,
            out_inference=out_inference,
            out_folder=out_folder,
            end_file=end_file,
            files=files,
        )
        # Set the random seed
        self.rng = get_or_create_rng(rng)

        # Initialize the fit parameters
        maxL = -self.inf  # initialization of the maximum  log-likelihood
        self.nodes = gdata.nodes
        conv = False  # initialization of the convergence flag
        best_loglik_values = []  # initialization of the log-likelihood values

        # Preprocess the data for fitting the models
        (
            data,
            data_T,
            data_T_vals,
            subs_nz,
            subs_nz_mask,
        ) = self._preprocess_data_for_fit(gdata.adjacency_tensor, mask)

        self.data = data
        self.data_T = data_T
        self.data_T_vals = data_T_vals
        self.mask = mask
        self.subs_nz = subs_nz
        self.subs_nz_mask = subs_nz_mask

        # Run the Expectation-Maximization (EM) algorithm for a specified number of realizations
        for r in range(self.num_realizations):
            # Initialize the parameters for the current realization
            (
                coincide,
                convergence,
                it,
                loglik,
                loglik_values,
            ) = self._initialize_realization()
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
        logging.debug(
            "Random number generator state: %s",
            self.rng.bit_generator.state["state"]["state"],
        )

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
        loglik_values: List[float] = []

        # Record the start time of the realization
        self.time_start = time.time()

        # Return the initial state of the realization
        return coincide, convergence, it, loglik, loglik_values

    def _preprocess_data_for_fit(
        self, data: GraphDataType, mask: MaskType
    ) -> Tuple[
        GraphDataType,
        np.ndarray,
        np.ndarray,
        tuple[int, int, int],
        tuple[int, int, int],
    ]:
        logging.debug("Preprocessing the data for fitting the models.")
        logging.debug("Data looks like: %s", data)

        data_T = np.einsum("aij->aji", data)
        data_T_vals = get_item_array_from_subs(data_T, data.nonzero())

        # pre-processing of the data to handle the sparsity
        data = preprocess_adjacency_tensor(data)
        data_T = preprocess_adjacency_tensor(data_T)

        # save the indexes of the nonzero entries
        subs_nz = tuple(self.get_data_nonzero(data))

        subs_nz_mask = mask.nonzero() if mask is not None else None

        return data, data_T, data_T_vals, subs_nz, subs_nz_mask

    def _initialize(self):
        """
        Random initialization of the parameters u, v, w, beta.
        """
        if not self.fix_pibr:
            self._randomize_pibr()

        if not self.fix_mupr:
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
            self._randomize_w()

    def _randomize_pibr(self):
        """
        Generate a random number in (0, 1.).
        """
        self.pibr = self.rng.random()

    def _randomize_mupr(self):
        """
        Generate a random number in (0, 1.).
        """
        self.mupr = self.rng.random()

    def _initialize_w(self, infile_name: str) -> None:  # type: ignore
        """
        Initialize affinity tensor w from file.

        Parameters
        ----------
        infile_name : str
                      Path of the input file.
        """

        with open(infile_name, "rb") as f:
            dfW = pd.read_csv(f, sep="\\s+", header=None)
            if self.assortative:
                self.w = np.diag(dfW)[np.newaxis, :].copy()
            else:
                self.w = dfW.values[np.newaxis, :, :]
        if not self.fix_w:
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

    def _update_cache(
        self,
        data: GraphDataType,
        data_T_vals: np.ndarray,
        subs_nz: Tuple[int, int, int],
    ) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda0_nz = super()._lambda_nz(subs_nz)
        if not self.assortative:
            self.lambda0_nzT = compute_mean_lambda0_nonzero(
                subs_nz, self.v, self.u, np.einsum("akq->aqk", self.w), self.assortative
            )
        else:
            self.lambda0_nzT = compute_mean_lambda0_nonzero(
                subs_nz, self.v, self.u, self.w, self.assortative
            )
        if self.flag_anomaly == True:
            self.Qij_dense, self.Qij_nz = self._QIJ(data, data_T_vals, subs_nz)
        self.M_nz = self.lambda0_nz
        self.M_nz[self.M_nz == 0] = 1

        if isinstance(data, np.ndarray):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, COO):
            self.data_M_nz = data.data / self.M_nz
            if self.flag_anomaly == True:
                self.data_M_nz_Q = data.data * (1 - self.Qij_nz) / self.M_nz
            else:
                self.data_M_nz_Q = data.data / self.M_nz

    def _QIJ(
        self,
        data: GraphDataType,
        data_T_vals: np.ndarray,
        subs_nz: SubsNzType,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        if isinstance(data, np.ndarray):
            nz_recon_I = np.power(1 - self.pibr, data[subs_nz])
        elif isinstance(data, COO):
            nz_recon_I = (
                self.mupr
                * poisson.pmf(data.data, self.pibr)
                * poisson.pmf(data_T_vals, self.pibr)
            )
            nz_recon_Id = nz_recon_I + (1 - self.mupr) * poisson.pmf(
                data.data, self.lambda0_nz
            ) * poisson.pmf(data_T_vals, self.lambda0_nzT)

            non_zeros = nz_recon_Id > 0
            nz_recon_I[non_zeros] /= nz_recon_Id[non_zeros]

        lambda0_ija = compute_mean_lambda0(self.u, self.v, self.w)
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

    def _update_em(
        self,
    ):
        """
        Update parameters via EM procedure.

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

        if not self.fix_communities:
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

        if not self.fix_w:
            if not self.assortative:
                d_w = self._update_W(self.subs_nz)
            else:
                d_w = self._update_W_assortative()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_w = 0

        if not self.fix_pibr:
            d_pibr = self._update_pibr(
                self.data,
                self.subs_nz,
                mask=self.mask,
                subs_nz_mask=self.subs_nz_mask,
            )
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)

        else:
            d_pibr = 0.0

        if not self.fix_mupr:
            d_mupr = self._update_mupr(
                mask=self.mask,
                subs_nz_mask=self.subs_nz_mask,
            )
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_mupr = 0.0

        return d_u, d_v, d_w, d_pibr, d_mupr

    def _update_pibr(
        self,
        data: GraphDataType,
        subs_nz: SubsNzType,
        mask: Optional[MaskType] = None,
        subs_nz_mask: Optional[SubsNzType] = None,
    ) -> float:
        """
        Update the anomaly parameter pi.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        subs_nz_mask : tuple
                       Indices of elements of data that are non-zero in the mask.

        Returns
        -------
        dist_pibr : float
                    Maximum distance between the old and the new anomaly parameter pi.
        """
        Adata = None
        if isinstance(data, np.ndarray):
            Adata = (data[subs_nz] * self.Qij_nz).sum()
        elif isinstance(data, COO):
            Adata = (data.data * self.Qij_nz).sum()
        else:
            log_and_raise_error(TypeError, "Data type not supported!")
        if mask is None:
            self.pibr = Adata / self.Qij_dense.sum()
        else:
            self.pibr = Adata / self.Qij_dense[subs_nz_mask].sum()

        dist_pibr = abs(self.pibr - self.pibr_old)
        self.pibr_old = np.copy(self.pibr)

        return dist_pibr

    def _update_mupr(
        self,
        mask: Optional[MaskType] = None,
        subs_nz_mask: Optional[SubsNzType] = None,
    ) -> float:
        """
        Update the prior mu parameter.

        Parameters
        ----------
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        subs_nz_mask : tuple
                       Indices of elements of data that are non-zero in the mask.

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
        self.mupr_old = np.copy(self.mupr)  # type: ignore

        return dist_mupr

    def _specific_update_U(self):
        """
        Specific logic for updating U in the AnomalyDetection class.
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

    def _specific_update_V(self):
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

    def _specific_update_W(self, subs_nz: SubsNzType, mask: MaskType = None):
        """
        Update affinity tensor.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """
        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum("Ik,Iq->Ikq", self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz_Q[:, np.newaxis, np.newaxis] * UV
        for _, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(
                subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L
            )

        self.w = self.ag - 1 + self.w * uttkrp_DKQ

        if self.flag_anomaly == True:
            mask = mask if mask is not None else 1
            UQk = np.einsum("aij,ik->ajk", mask * (1 - self.Qij_dense), self.u)
            Z = np.einsum("ajk,jq->akq", UQk, self.v)
        else:  # flag_anomaly == False
            Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))[
                np.newaxis, :, :
            ]
        Z += self.bg

        non_zeros = Z > 0
        self.w[Z == 0] = 0.0
        self.w[non_zeros] /= Z[non_zeros]

    def _update_W(self, subs_nz: SubsNzType) -> float:
        # a generic function here that will do what each class needs
        self._specific_update_W(subs_nz)

        dist, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

        return dist

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

    def _update_membership(
        self,
        subs_nz: ArraySequence,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        m: int,
    ) -> np.ndarray:
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
        return self._ELBO(self.data, self.mask, self.subs_nz_mask)

    def _ELBO(
        self,
        data: GraphDataType,
        mask: Optional[MaskType] = None,
        subs_nz_mask: Optional[SubsNzType] = None,
    ) -> float:
        """
        Compute the Evidence Lower BOund (ELBO) of the data.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        subs_nz_mask : tuple
                       Indices of elements of data that are non-zero in the mask.

        Returns
        -------
        l : float
            The computed ELBO value.
        """

        self.lambda0_ija = compute_mean_lambda0(self.u, self.v, self.w)

        if mask is not None:
            if isinstance(data, COO):
                Adense = data.todense()
            else:
                raise ValueError("Mask is not None but data is not a COO tensor.")

        if not self.flag_anomaly:
            l = (data.data * np.log(self.lambda0_ija[data.coords] + EPS_)).sum()
            l -= (
                self.lambda0_ija.sum()
                if mask is None
                else self.lambda0_ija[subs_nz_mask].sum()
            )
            return l
        else:
            l = 0.0

            # Term containing Q, pi and A
            l -= self.pibr * self.Qij_dense.sum()

            if self.pibr >= 0:
                if mask is None:
                    coords_tuple = tuple(data.coords[i] for i in range(3))
                    l += (
                        np.log(self.pibr + EPS_)
                        * (self.Qij_dense[coords_tuple] * data.data).sum()
                    )
                else:
                    subs_nz = np.logical_and(mask > 0, Adense > 0)
                    l += (
                        np.log(self.pibr + EPS_)
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
                self.Qij_dense[non_zeros] * np.log(self.Qij_dense[non_zeros] + EPS_)
            ).sum()
            l -= (
                (1 - self.Qij_dense)[non_zeros1]
                * np.log((1 - self.Qij_dense)[non_zeros1] + EPS_)
            ).sum()

            # Term containing Q, M and A
            if mask is None:
                l -= ((1 - self.Qij_dense) * self.lambda0_ija).sum()
                coords_tuple = tuple(data.coords[i] for i in range(3))
                l += (
                    ((1 - self.Qij_dense)[coords_tuple])
                    * data.data
                    * np.log(self.lambda0_ija[coords_tuple] + EPS_)
                ).sum()

                # Term containing Q and mu
                if 1 - self.mupr >= 0:
                    l += np.log(1 - self.mupr + EPS_) * (1 - self.Qij_dense).sum()
                if self.mupr >= 0:
                    l += np.log(self.mupr + EPS_) * (self.Qij_dense).sum()
            else:
                l -= (
                    (1 - self.Qij_dense[subs_nz_mask]) * self.lambda0_ija[subs_nz_mask]
                ).sum()
                subs_nz = np.logical_and(mask > 0, Adense > 0)
                l += (
                    ((1 - self.Qij_dense)[subs_nz])
                    * data.data
                    * np.log(self.lambda0_ija[subs_nz] + EPS_)
                ).sum()

                if 1 - self.mupr > 0:
                    l += (
                        np.log(1 - self.mupr + EPS_)
                        * (1 - self.Qij_dense)[subs_nz_mask].sum()
                    )
                if self.mupr > 0:
                    l += np.log(self.mupr + EPS_) * (self.Qij_dense[subs_nz_mask]).sum()

            if self.ag > 1.0:
                l += (self.ag - 1) * np.log(self.u + EPS_).sum()
                l += (self.ag - 1) * np.log(self.v + EPS_).sum()
            if self.bg > 0.0:
                l -= self.bg * self.u.sum()
                l -= self.bg * self.v.sum()

            if np.isnan(l):
                log_and_raise_error(ValueError, "ELBO is NaN!")
            return l

    def _log_realization_info(
        self,
        r: int,
        loglik: float,
        maxL: float,
        final_it: int,
        time_start: float,
        convergence: bool,
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
