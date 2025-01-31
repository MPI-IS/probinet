"""
Class definition of CRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and reciprocity value
:cite:`safdari2021generative`.
"""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy import dtype, ndarray
from sparse import COO

from ..evaluation.expectation_computation import compute_mean_lambda0
from ..input.preprocessing import preprocess_adjacency_tensor
from ..types import ArraySequence, EndFileType, FilesType, GraphDataType, MaskType
from ..utils.matrix_operations import sp_uttkrp, sp_uttkrp_assortative
from ..utils.tools import (
    get_item_array_from_subs,
    get_or_create_rng,
    log_and_raise_error,
)
from .base import ModelBase, ModelUpdateMixin
from .classes import GraphData
from .constants import OUTPUT_FOLDER


class CRep(ModelBase, ModelUpdateMixin):
    """
    Class to perform inference in networks with reciprocity.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        num_realizations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_iter=max_iter, num_realizations=num_realizations, **kwargs)
        self.__doc__ = ModelBase.__doc__

        # Initialize other attributes
        self.eta_f = 0.0

    def load_data(self, **kwargs: Any):
        # Check that the parameters are correct
        self.check_params_to_load_data(**kwargs)
        # Load and return the data
        return super().load_data(**kwargs)

    def check_params_to_load_data(self, **kwargs):
        if not kwargs["binary"]:
            log_and_raise_error(
                ValueError, "CRep requires the parameter `binary` to be True."
            )
        if not kwargs["noselfloop"]:
            log_and_raise_error(
                ValueError, "CRep requires the parameter `noselfloop` to be True."
            )
        if kwargs["undirected"]:
            log_and_raise_error(
                ValueError, "CRep requires the parameter `undirected` to be False."
            )

    def _check_fit_params(
        self,
        **kwargs: Any,
    ) -> None:
        # Call the check_fit_params method from the parent class
        super()._check_fit_params(**kwargs)

        self._validate_eta0(kwargs["eta0"])
        self.eta0 = kwargs["eta0"]

        self.constrained = kwargs.get("constrained", True)

        # Parameters for the initialization of the models
        self.use_unit_uniform = True
        self.normalize_rows = True

        self._validate_undirected_eta()

        if self.initialization == 1:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)

    def fit(
        self,
        gdata: GraphData,
        K: int = 3,
        mask: Optional[MaskType] = None,
        initialization: int = 0,
        eta0: Optional[float] = None,
        undirected: bool = False,
        assortative: bool = True,
        constrained: bool = True,
        out_inference: bool = True,
        fix_eta: bool = False,
        fix_w: bool = False,
        out_folder: Path = OUTPUT_FOLDER,
        end_file: Optional[EndFileType] = None,
        files: Optional[FilesType] = None,
        rng: Optional[np.random.Generator] = None,
        **_kwargs: Any,
    ) -> tuple[
        ndarray[Any, dtype[np.float64]],
        ndarray[Any, dtype[np.float64]],
        ndarray[Any, dtype[np.float64]],
        float,
        float,
    ]:
        """
        Fit the CRep models to the given data using the EM algorithm.

        Parameters
        ----------
        data
            Graph adjacency tensor.
        data_T
            Transposed graph adjacency tensor.
        data_T_vals
            Array with values of entries A[j, i] given non-zero entry (i, j).
        nodes
            List of node IDs.
        K
            Number of communities, by default 3.
        mask
            Mask for selecting the held-out set in the adjacency tensor in case of cross-validation, by default None.
        initialization
            Initialization method for the models parameters, by default 0.
        eta0
            Initial value of the reciprocity coefficient, by default None.
        undirected
            Flag to specify if the graph is undirected, by default False.
        assortative
            Flag to specify if the graph is assortative, by default True.
        constrained
            Flag to specify if the models is constrained, by default True.
        out_inference
            Flag to specify if inference results should be evaluation, by default True.
        out_folder
            Output folder for inference results, by default "outputs/".
        end_file
            Suffix for the evaluation file, by default "_CRep".
        fix_eta
            Flag to specify if the eta parameter should be fixed, by default False.
        files
            Path to the file for initialization, by default "".
        rng
            Random number generator.

        Returns
        -------
        u_f
            Out-going membership matrix.
        v_f
            In-coming membership matrix.
        w_f
            Affinity tensor.
        eta_f
            Reciprocity coefficient.
        maxL
            Maximum pseudo log-likelihood.
        """

        self._check_fit_params(
            data=gdata.adjacency_tensor,
            K=K,
            initialization=initialization,
            eta0=eta0,
            undirected=undirected,
            assortative=assortative,
            constrained=constrained,
            out_inference=out_inference,
            out_folder=out_folder,
            end_file=end_file,
            fix_eta=fix_eta,
            fix_w=fix_w,
            files=files,
        )
        # Set the random seed
        self.rng = get_or_create_rng(rng)

        # Initialize the fit parameters
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood
        self.nodes = gdata.nodes
        conv = False  # initialization of the convergence flag
        best_loglik_values = []  # initialization of the log-likelihood values

        # Preprocess the data for fitting the models
        E, data, data_T, data_T_vals, subs_nz = self._preprocess_data_for_fit(
            gdata.adjacency_tensor, gdata.transposed_tensor, gdata.data_values
        )
        # Set the preprocessed data and other related variables as attributes of the class instance
        self.data = data
        self.data_T = data_T
        self.data_T_vals = data_T_vals
        self.subs_nz = subs_nz
        self.denominator = E
        self.mask = mask

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
                super()._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
                self.best_r = r
                if self.flag_conv == "log":
                    best_loglik_values = list(loglik_values)

            # Log the current realization number, log-likelihood, number of iterations, and elapsed time
            self._log_realization_info(
                r, loglik, self.final_it, self.time_start, convergence
            )

        # end cycle over realizations

        # Store the maximum pseudo log-likelihood
        self.maxPSL = maxL

        # Evaluate the results of the fitting process
        self._evaluate_fit_results(self.maxPSL, conv, best_loglik_values)

        # Return the final parameters and the maximum log-likelihood
        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _initialize_realization(self) -> Tuple[int, bool, int, float, List[float]]:
        """
        This method initializes the parameters for each realization of the EM algorithm.
        It also sets up local variables for convergence checking.
        """

        # Log the current state of the random number generator
        logging.debug(
            "Random number generator seed: %s",
            self.rng.bit_generator.state["state"]["state"],
        )

        # Initialize the parameters for the current realization
        super()._initialize()

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
        self,
        data: GraphDataType,
        data_T: Union[GraphDataType, None],
        data_T_vals: Union[np.ndarray, None],
    ) -> tuple[float, COO | ndarray, COO | ndarray, ndarray | None, tuple]:
        """
        Preprocess the data for fitting the models.

        Parameters
        ----------
        data : COO, np.ndarray
            The input data tensor.
        data_T : COO, np.ndarray or None
            The transposed input data tensor. If None, it will be calculated from the input data tensor.
        data_T_vals : np.ndarray or None
            The values of the non-zero entries in the transposed input data tensor. If None, it will be calculated from data_T.

        Returns
        -------
        tuple
            A tuple containing the preprocessed data tensor, the values of the non-zero entries in the transposed data tensor,
            and the indices of the non-zero entries in the data tensor.
        """

        # If data_T is not provided, calculate it from the input data tensor
        if data_T is None:
            E = self.get_data_sum(
                data
            )  # weighted sum of edges (needed in the denominator of eta)
            data_T = np.einsum("aij->aji", data)
            data_T_vals = get_item_array_from_subs(data_T, self.get_data_nonzero(data))
            # Pre-process the data to handle the sparsity
            data = preprocess_adjacency_tensor(data)
            data_T = preprocess_adjacency_tensor(data_T)
        else:
            E = self.get_data_sum(data)

        # Save the indices of the non-zero entries
        subs_nz = self.get_data_nonzero(data)

        return E, data, data_T, data_T_vals, subs_nz

    def compute_likelihood(self) -> float:
        """
        Compute the pseudo log-likelihood of the data.

        Returns
        -------
        loglik : float
                 Pseudo log-likelihood value.
        """
        return self._ps_likelihood(self.data, self.data_T, self.mask)

    def _update_cache(
        self,
        data: GraphDataType,
        data_T_vals: np.ndarray,
        subs_nz: ArraySequence,
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
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals
        self.M_nz[self.M_nz == 0] = 1
        self.data_M_nz = self.get_data_values(data) / self.M_nz
        self.data_M_nz[self.M_nz == 0] = 0

    def _update_em(self):
        """
        Update parameters via EM procedure.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        denominator : float
                      Denominator used in the update of the eta parameter.

        """

        if not self.fix_eta:
            d_eta = self._update_eta(
                self.data, self.data_T_vals, denominator=self.denominator
            )
        else:
            d_eta = 0.0
        self._update_cache(self.data, self.data_T_vals, self.subs_nz)

        if not self.fix_communities:
            d_u = self._update_U()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_u = 0.0

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            if not self.fix_communities:
                d_v = self._update_V()
                self._update_cache(self.data, self.data_T_vals, self.subs_nz)
            else:
                d_v = 0.0

        if not self.fix_w:
            if not self.assortative:
                d_w = self._update_W()
            else:
                d_w = self._update_W_assortative()
            self._update_cache(self.data, self.data_T_vals, self.subs_nz)
        else:
            d_w = 0

        self.delta_u = d_u
        self.delta_v = d_v
        self.delta_w = d_w
        self.delta_eta = d_eta

    def _update_eta(
        self,
        data: GraphDataType,
        data_T_vals: np.ndarray,
        denominator: Optional[float] = None,
    ) -> float:
        """
        Update reciprocity coefficient eta.

        Parameters
        ----------
        data : GraphDataType
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

        dist_eta = abs(self.eta - self.eta_old)  # type: ignore
        self.eta_old = float(self.eta)

        return dist_eta

    def _specific_update_U(self):
        self.u = self.u_old * self._update_membership(self.subs_nz, 1)  # type: ignore

        if not self.constrained:
            Du = np.einsum("iq->q", self.v)
            if not self.assortative:
                w_k = np.einsum("akq->kq", self.w)
                Z_uk = np.einsum("q,kq->k", Du, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_uk = np.einsum("k,k->k", Du, w_k)
            non_zeros = Z_uk > 0.0
            self.u[:, Z_uk == 0] = 0.0
            self.u[:, non_zeros] /= Z_uk[np.newaxis, non_zeros]
        else:
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.0  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))  # type: ignore
        self.u_old = np.copy(self.u)

        return dist_u

    def _specific_update_V(self):
        self.v *= self._update_membership(self.subs_nz, 2)

        if not self.constrained:
            Dv = np.einsum("iq->q", self.u)
            if not self.assortative:
                w_k = np.einsum("aqk->qk", self.w)
                Z_vk = np.einsum("q,qk->k", Dv, w_k)
            else:
                w_k = np.einsum("ak->k", self.w)
                Z_vk = np.einsum("k,k->k", Dv, w_k)
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.0
            self.v[:, non_zeros] /= Z_vk[np.newaxis, non_zeros]
        else:
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

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

        Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))[
            np.newaxis, :, :
        ]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

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

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

    def _update_membership(self, subs_nz: ArraySequence, m: int) -> np.ndarray:
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
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, self.u, self.v, self.w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(
                self.data_M_nz, subs_nz, m, self.u, self.v, self.w
            )

        return uttkrp_DK

    def _ps_likelihood(
        self,
        data: GraphDataType,
        data_T: COO,
        mask: Optional[MaskType] = None,
    ) -> float:
        """
        Compute the pseudo log-likelihood of the data.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        data_T : GraphDataType
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of
               cross-validation.

        Returns
        -------
        loglik : float
            Pseudo log-likelihood value.
        """
        # Compute the mean lambda0 for all entries
        self.lambda0_ija = compute_mean_lambda0(self.u, self.v, self.w)

        # Get the non-zero entries of the mask
        sub_mask_nz = mask.nonzero() if mask is not None else None

        if sub_mask_nz is not None:
            loglik = (
                -self.lambda0_ija[sub_mask_nz].sum()
                - self.eta * self.get_data_toarray(data_T)[sub_mask_nz].sum()
            )
        else:
            loglik = -self.lambda0_ija.sum() - self.eta * self.get_data_sum(data_T)

        logM = np.log(self.M_nz)
        Alog = self.get_data_values(data) * logM

        loglik += Alog.sum()

        if np.isnan(loglik):
            log_and_raise_error(ValueError, "PSLikelihood is NaN!!!!")
        return loglik

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
        source_var = getattr(self, f"eta{source_suffix}")
        setattr(self, f"eta{target_suffix}", float(source_var))
