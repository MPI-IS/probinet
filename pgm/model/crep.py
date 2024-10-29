"""
Class definition of CRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and reciprocity value.
"""

import logging
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

from numpy import dtype, ndarray
import numpy as np
from sparse import COO

from ..input.preprocessing import preprocess
from ..input.tools import (
    get_item_array_from_subs, inherit_docstring, log_and_raise_error, sp_uttkrp,
    sp_uttkrp_assortative)
from ..output.evaluate import lambda0_full
from .base import ModelBase, ModelUpdateMixin
from .classes import GraphData


class CRep(ModelBase, ModelUpdateMixin):
    """
    Class to perform inference in networks with reciprocity.
    """

    @inherit_docstring(ModelBase)
    def __init__(
        self,
        max_iter: int = 1000,
        num_realizations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_iter=max_iter, num_realizations=num_realizations, **kwargs)

        # Initialize other attributes
        self.eta_f = 0.0

    def load_data(self, **kwargs: Any):
        # Check that the parameters are correct
        self.check_params_to_load_data(**kwargs)
        # Load and return the data
        return super().load_data(**kwargs)

    def check_params_to_load_data(self, **kwargs):
        if not kwargs["binary"]:
            log_and_raise_error(ValueError, "CRep requires the parameter `binary` to be True.")
        if not kwargs["noselfloop"]:
            log_and_raise_error(ValueError, "CRep requires the parameter `noselfloop` to be True.")
        if kwargs["undirected"]:
            log_and_raise_error(ValueError, "CRep requires the parameter `undirected` to be False.")

    def _check_fit_params(
        self,
        **kwargs: Any,
    ) -> None:

        message = "The initialization parameter can be either 0, 1, 2 or 3."

        super()._check_fit_params(message=message, **kwargs)

        self.constrained = kwargs.get("constrained", True)

        # Parameters for the initialization of the model
        self.use_unit_uniform = True
        self.normalize_rows = True

        if self.initialization == 1:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)

    def fit(
        self,
        gdata: GraphData,
        rseed: int = 0,
        K: int = 3,
        mask: Optional[np.ndarray] = None,
        initialization: int = 0,
        eta0: Union[float, None] = None,
        undirected: bool = False,
        assortative: bool = True,
        constrained: bool = True,
        out_inference: bool = True,
        out_folder: Path = Path("outputs"),
        end_file: str = None,
        fix_eta: bool = False,
        files: str = None,
        **_kwargs: Any,
    ) -> tuple[
        ndarray[Any, dtype[np.float64]],
        ndarray[Any, dtype[np.float64]],
        ndarray[Any, dtype[np.float64]],
        float,
        float,
    ]:
        """
        Fit the CRep model to the given data using the EM algorithm.

        Parameters
        ----------
        data : Union[COO, np.ndarray]
            Graph adjacency tensor.
        data_T : COO
            Transposed graph adjacency tensor.
        data_T_vals : np.ndarray
            Array with values of entries A[j, i] given non-zero entry (i, j).
        nodes : List[Any]
            List of node IDs.
        rseed : int, optional
            Random seed, by default 0.
        K : int, optional
            Number of communities, by default 3.
        mask : Optional[np.ndarray], optional
            Mask for selecting the held-out set in the adjacency tensor in case of cross-validation, by default None.
        initialization : int, optional
            Initialization method for the model parameters, by default 0.
        eta0 : Union[float, None], optional
            Initial value of the reciprocity coefficient, by default None.
        undirected : bool, optional
            Flag to specify if the graph is undirected, by default False.
        assortative : bool, optional
            Flag to specify if the graph is assortative, by default True.
        constrained : bool, optional
            Flag to specify if the model is constrained, by default True.
        out_inference : bool, optional
            Flag to specify if inference results should be output, by default True.
        out_folder : str, optional
            Output folder for inference results, by default "outputs/".
        end_file : str, optional
            Suffix for the output file, by default "_CRep".
        fix_eta : bool, optional
            Flag to specify if the eta parameter should be fixed, by default False.
        files : str, optional
            Path to the file for initialization, by default "".

        Returns
        -------
        tuple
            A tuple containing:
            -u_f : ndarray
            Out-going membership matrix.
        -v_f : ndarray
            In-coming membership matrix.
        -w_f : ndarray
            Affinity tensor.
        -eta_f : float
            Reciprocity coefficient.
        -maxL : float
            Maximum pseudo log-likelihood.
        """

        self._check_fit_params(
            data=gdata.incidence_tensor,
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
            files=files,
        )
        # Set the random seed
        self.rseed = rseed
        logging.debug("Fixing random seed to: %s", self.rseed)
        self.rng = np.random.RandomState(self.rseed)

        # Initialize the fit parameters
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum pseudo log-likelihood
        self.nodes = gdata.nodes

        # Preprocess the data for fitting the model
        E, data, data_T, data_T_vals, subs_nz = self._preprocess_data_for_fit(
            gdata.incidence_tensor, gdata.transposed_tensor, gdata.data_values
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
        logging.debug("Random number generator seed: %s", self.rng.get_state()[1][0])  # type: ignore

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
        data: Union[COO, np.ndarray],
        data_T: Union[COO, np.ndarray, None],
        data_T_vals: Union[np.ndarray, None],
    ) -> Tuple[int, Any, Any, np.ndarray, Tuple[np.ndarray]]:
        """
        Preprocess the data for fitting the model.

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
            E = np.sum(data)  # weighted sum of edges (needed in the denominator of eta)
            data_T = np.einsum("aij->aji", data)
            data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
            # Pre-process the data to handle the sparsity
            data = preprocess(data)
            data_T = preprocess(data_T)
        else:
            E = np.sum(data.data)

        # Save the indices of the non-zero entries
        if isinstance(data, np.ndarray):
            subs_nz = data.nonzero()
        elif isinstance(data, COO):
            subs_nz = data.coords

        return E, data, data_T, data_T_vals, subs_nz  # type: ignore

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
        data: Union[COO, np.ndarray],
        data_T_vals: np.ndarray,
        subs_nz: Tuple[np.ndarray],
    ) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : Union[COO, np.ndarray]
               Graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda0_nz = super()._lambda_nz(subs_nz)
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals
        self.M_nz[self.M_nz == 0] = 1
        if isinstance(data, np.ndarray):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, COO):
            self.data_M_nz = data.data / self.M_nz
        self.data_M_nz[self.M_nz == 0] = 0

    def _update_em(self):
        """
        Update parameters via EM procedure.

        Parameters
        ----------
        data : Union[COO, np.ndarray]
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
        data: Union[COO, np.ndarray],
        data_T_vals: np.ndarray,
        denominator: Optional[float] = None,
    ) -> float:
        """
        Update reciprocity coefficient eta.

        Parameters
        ----------
        data : Union[COO, np.ndarray]
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

    def _update_membership(self, subs_nz: Tuple[np.ndarray], m: int) -> np.ndarray:
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
        data: Union[COO, np.ndarray],
        data_T: COO,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the pseudo log-likelihood of the data.

        Parameters
        ----------
        data : Union[COO, np.ndarray]
               Graph adjacency tensor.
        data_T : Union[COO, np.ndarray]
                 Graph adjacency tensor (transpose).
        mask : ndarray
               Mask for selecting the held out set in the adjacency tensor in case of
               cross-validation.

        Returns
        -------
        loglik : float
            Pseudo log-likelihood value.
        """

        self.lambda0_ija = lambda0_full(self.u, self.v, self.w)

        if mask is not None:
            sub_mask_nz = mask.nonzero()
            if isinstance(data, np.ndarray):
                loglik = (
                    -self.lambda0_ija[sub_mask_nz].sum()
                    - self.eta * data_T[sub_mask_nz].sum()
                )
            elif isinstance(data, COO):
                loglik = (
                    -self.lambda0_ija[sub_mask_nz].sum()
                    - self.eta * data_T.toarray()[sub_mask_nz].sum()
                )
        else:
            if isinstance(data, np.ndarray):
                loglik = -self.lambda0_ija.sum() - self.eta * data_T.sum()
            elif isinstance(data, COO):
                loglik = -self.lambda0_ija.sum() - self.eta * data_T.data.sum()
        logM = np.log(self.M_nz)
        if isinstance(data, np.ndarray):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, COO):
            Alog = data.data * logM

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
