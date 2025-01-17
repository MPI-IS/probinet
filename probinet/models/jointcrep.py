"""
Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and a pair interaction value
:cite:`contisciani2022community`.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..evaluation.expectation_computation import compute_mean_lambda0
from ..input.preprocessing import preprocess_adjacency_tensor
from ..types import EndFileType, FilesType, GraphDataType
from ..utils.matrix_operations import sp_uttkrp, sp_uttkrp_assortative, transpose_tensor
from ..utils.tools import (
    check_symmetric,
    get_item_array_from_subs,
    get_or_create_rng,
    log_and_raise_error,
)
from .base import ModelBase, ModelUpdateMixin
from .classes import GraphData
from .constants import K_DEFAULT, OUTPUT_FOLDER


class JointCRep(ModelBase, ModelUpdateMixin):
    """
    Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.__doc__ = ModelBase.__init__.__doc__

    def _check_fit_params(
        self,
        **kwargs: Any,
    ) -> None:
        # Call the check_fit_params method from the parent class
        super()._check_fit_params(**kwargs)

        self._validate_eta0(kwargs["eta0"])
        self.eta0 = kwargs["eta0"]

        # Parameters for the initialization of the models
        self.normalize_rows = False
        self.use_unit_uniform = False

        self._validate_undirected_eta()

        if self.initialization == 1:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)
            dfW = self.theta["w"]
            self.L = dfW.shape[0]
            self.K = dfW.shape[1]

        if self.fix_eta:
            self.eta = self.eta_old = self.eta_f = self.eta0  # type: ignore

    def fit(
        self,
        gdata: GraphData,
        K: int = K_DEFAULT,
        initialization: int = 0,
        eta0: Optional[float] = None,
        undirected: bool = False,
        assortative: bool = True,
        fix_eta: bool = False,
        fix_communities: bool = False,
        fix_w: bool = False,
        use_approximation: bool = False,
        out_inference: bool = True,
        out_folder: Path = OUTPUT_FOLDER,
        end_file: Optional[EndFileType] = None,
        files: Optional[FilesType] = None,
        rng: Optional[np.random.Generator] = None,
        **_kwargs: Any,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        float,
        float,
    ]:
        """
        Model directed networks by using a probabilistic generative model based on a Bivariate
        Bernoulli distribution that assumes community parameters and a pair interaction
        coefficient as latent variables. The inference is performed via the EM algorithm.

        Parameters
        ----------
        gdata
            Graph adjacency tensor.
        K
            Number of communities, by default 3.
        initialization
            Indicator for choosing how to initialize u, v, and w. If 0, they will be generated randomly;
            1 means only the affinity matrix w will be uploaded from file; 2 implies the membership
            matrices u and v will be uploaded from file, and 3 all u, v, and w will be initialized
            through an input file, by default 0.
        eta0
            Initial value for the reciprocity coefficient, by default None.
        undirected
            Flag to call the undirected network, by default False.
        assortative
            Flag to call the assortative network, by default True.
        fix_eta
            Flag to fix the eta parameter, by default False.
        fix_communities
            Flag to fix the community memberships, by default False.
        fix_w
            Flag to fix the affinity tensor, by default False.
        use_approximation
            Flag to use approximation in updates, by default False.
        out_inference
            Flag to evaluate inference results, by default True.
        out_folder
            Output folder for inference results, by default OUTPUT_FOLDER.
        end_file
            Suffix for the evaluation file, by default None.
        files
            Path to the file for initialization, by default None.
        rng
            Random number generator, by default None.

        Returns
        -------

        u_f
            Out-going membership matrix.
        v_f
            In-coming membership matrix.
        w_f
            Affinity tensor.
        eta_f
            Pair interaction coefficient.
        maxL
            Maximum log-likelihood.
        """

        # Check the parameters for fitting the models
        self._check_fit_params(
            data=gdata.adjacency_tensor,
            K=K,
            initialization=initialization,
            eta0=eta0,
            undirected=undirected,
            assortative=assortative,
            use_approximation=use_approximation,
            fix_eta=fix_eta,
            fix_communities=fix_communities,
            fix_w=fix_w,
            files=files,
            out_inference=out_inference,
            out_folder=out_folder,
            end_file=end_file,
        )

        # Set the random seed
        self.rng = get_or_create_rng(rng)

        # Initialize the fit parameters
        self.initialization = initialization
        maxL = -self.inf  # initialization of the maximum log-likelihood
        self.nodes = gdata.nodes
        conv = False  # initialization of the convergence flag
        best_loglik_values = []  # initialization of the log-likelihood values

        # Preprocess the data for fitting the models
        data, data_T_vals, subs_nz = self._preprocess_data_for_fit(
            gdata.adjacency_tensor, gdata.transposed_tensor, gdata.data_values
        )

        # Calculate the sum of the product of non-zero values in data and data_T
        self.AAtSum = (self.get_data_values(data) * data_T_vals).sum()

        # Store the preprocessed data and the indices of its non-zero elements
        self.data = data
        self.subs_nz = subs_nz

        # Run the Expectation-Maximization (EM) algorithm for a specified number of realizations
        for r in range(self.num_realizations):
            # Initialize the parameters for the current realization
            (
                coincide,
                convergence,
                it,
                loglik,
                loglik_values,
            ) = self._initialize_realization(data, subs_nz)

            # Update the parameters for the current realization
            it, loglik, coincide, convergence, loglik_values = self._update_realization(
                r, it, loglik, coincide, convergence, loglik_values
            )

            # If the current log-likelihood is greater than the maximum log-likelihood so far,
            # update the optimal parameters and the maximum log-likelihood
            if maxL < loglik and convergence:
                logging.debug("Better log-likelihood found in realization %s.", r)
                logging.debug("Updating optimal parameters.")
                super()._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
                self.best_r = r

                # If the convergence criterion is 'log', store the log-likelihood values
                if self.flag_conv == "log":
                    best_loglik_values = list(loglik_values)

            # Log the current realization number, log-likelihood, number of iterations, and elapsed time
            self._log_realization_info(
                r, loglik, self.final_it, self.time_start, convergence
            )

        # End cycle over realizations

        # Store the maximum log-likelihood
        self.maxL = maxL

        # Evaluate the results of the fitting process
        self._evaluate_fit_results(self.maxL, conv, best_loglik_values)

        # Return the final parameters and the maximum log-likelihood
        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL  # type: ignore

    def _initialize_realization(self, data, subs_nz):
        """
        This method initializes the parameters for each realization of the EM algorithm.
        It also sets up local variables for convergence checking.
        """
        # Log the current state of the random number generator
        logging.debug(
            "Random number generator seed: %s",
            self.rng.bit_generator.state["state"]["state"],
        )

        # Call the _initialize method from the parent class to initialize the parameters for the current realization
        super()._initialize()

        # Call the _update_old_variables method from the parent class to update the old variables for the current realization
        super()._update_old_variables()

        # Update the cache used in the EM update
        self._update_cache(data, subs_nz)

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

    def _preprocess_data_for_fit(
        self,
        data: GraphDataType,
        data_T: Union[GraphDataType, None],
        data_T_vals: Union[np.ndarray, None],
    ) -> Tuple[GraphDataType, np.ndarray, tuple]:
        """
        Preprocess the data for fitting the models.

        Parameters
        ----------
        data : COO, np.ndarray
            The input data tensor.
        data_T : COO, np.ndarray, None
            The transposed input data tensor. If None, it will be calculated from the input data tensor.
        data_T_vals : np.ndarray, None
            The values of the non-zero entries in the transposed input data tensor. If None, it will be calculated from data_T.

        Returns
        -------
        tuple
            A tuple containing the preprocessed data tensor, the values of the non-zero entries in the transposed data tensor,
            and the indices of the non-zero entries in the data tensor.
        """

        # If data_T is not provided, calculate it from the input data tensor
        if data_T is None:
            data_T = np.einsum("aij->aji", data)
            data_T_vals = get_item_array_from_subs(data_T, self.get_data_nonzero(data))
            # Pre-process the data to handle the sparsity
            data = preprocess_adjacency_tensor(data)

        # Save the indices of the non-zero entries
        subs_nz = self.get_data_nonzero(data)

        return data, data_T_vals, subs_nz  # type: ignore

    def compute_likelihood(self) -> float:
        """
        Compute the log-likelihood of the data.

        Returns
        -------
        loglik : float
            Log-likelihood value.
        """
        return self._likelihood()

    def _update_cache(self, data: GraphDataType, subs_nz: tuple) -> None:
        """
        Update the cache used in the em_update.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.lambda_aij = compute_mean_lambda0(
            self.u, self.v, self.w
        )  # full matrix lambda

        self.lambda_nz = super()._lambda_nz(
            subs_nz
        )  # matrix lambda for non-zero entries
        lambda_zeros = self.lambda_nz == 0
        self.lambda_nz[lambda_zeros] = 1  # still good because with np.log(1)=0
        self.data_M_nz = self.get_data_values(data) / self.lambda_nz
        self.data_M_nz[lambda_zeros] = 0  # to use in the updates

        self.den_updates = 1 + self.eta * self.lambda_aij  # to use in the updates
        if not self.use_approximation:
            self.lambdalambdaT = np.einsum(
                "aij,aji->aij", self.lambda_aij, self.lambda_aij
            )  # to use in Z and eta
            self.Z = self._calculate_Z()

    def _calculate_Z(self) -> np.ndarray:
        """
        Compute the normalization constant of the Bivariate Bernoulli distribution.

        Returns
        -------
        Z : ndarray
            Normalization constant Z of the Bivariate Bernoulli distribution.
        """

        Z = (
            self.lambda_aij
            + transpose_tensor(self.lambda_aij)
            + self.eta * self.lambdalambdaT
            + 1
        )
        for _, z in enumerate(Z):
            assert check_symmetric(z)

        return Z

    def _update_em(self) -> tuple:
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
        d_eta : float
                Maximum distance between the old and the new pair interaction coefficient eta.
        """

        if not self.fix_communities:
            if self.use_approximation:
                d_u = self._update_U_approx()
            else:
                d_u = self._update_U()
            self._update_cache(self.data, self.subs_nz)
        else:
            d_u = 0.0

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
            self._update_cache(self.data, self.subs_nz)
        else:
            if not self.fix_communities:
                if self.use_approximation:
                    d_v = self._update_V_approx()
                else:
                    d_v = self._update_V()
                self._update_cache(self.data, self.subs_nz)
            else:
                d_v = 0.0

        if not self.fix_w:
            if not self.assortative:
                if self.use_approximation:
                    d_w = self._update_W_approx()
                else:
                    d_w = self._update_W()
            else:
                if self.use_approximation:
                    d_w = self._update_W_assortative_approx()
                else:
                    d_w = self._update_W_assortative()
            self._update_cache(self.data, self.subs_nz)
        else:
            d_w = 0.0

        if not self.fix_eta:
            self.lambdalambdaT = np.einsum(
                "aij,aji->aij", self.lambda_aij, self.lambda_aij
            )  # to use in Z and eta
            if self.use_approximation:
                d_eta = self._update_eta_approx()
            else:
                d_eta = self._update_eta()
            self._update_cache(self.data, self.subs_nz)
        else:
            d_eta = 0.0

        return d_u, d_v, d_w, d_eta

    def _update_U_approx(self) -> float:
        """
        Update out-going membership matrix by using an approximation.

        Returns
        -------
        dist_u : float
                 Maximum distance between the old and the new membership matrix u.
        """

        self.u *= self._update_membership(self.subs_nz, 1)

        if not self.assortative:
            VW = np.einsum("jq,akq->ajk", self.v, self.w)
        else:
            VW = np.einsum("jk,ak->ajk", self.v, self.w)
        den = np.einsum("aji,ajk->ik", self.den_updates, VW)

        non_zeros = den > 0.0
        self.u[den == 0] = 0.0
        self.u[non_zeros] /= den[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.0  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _specific_update_U(self):
        self.u *= self._update_membership(self.subs_nz, 1)

        if not self.assortative:
            VW = np.einsum("jq,akq->ajk", self.v, self.w)
        else:
            VW = np.einsum("jk,ak->ajk", self.v, self.w)
        VWL = np.einsum("aji,ajk->aijk", self.den_updates, VW)
        den = np.einsum("aijk,aij->ik", VWL, 1.0 / self.Z)

        non_zeros = den > 0.0
        self.u[den == 0] = 0.0
        self.u[non_zeros] /= den[non_zeros]

    def _update_V_approx(self) -> float:
        """
        Update in-coming membership matrix by using an approximation.
        Same as _update_U but with:
        data <-> data_T
        w <-> w_T
        u <-> v

        Returns
        -------
        dist_v : float
                 Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(self.subs_nz, 2)

        if not self.assortative:
            UW = np.einsum("jq,aqk->ajk", self.u, self.w)
        else:
            UW = np.einsum("jk,ak->ajk", self.u, self.w)
        den = np.einsum("aij,ajk->ik", self.den_updates, UW)

        non_zeros = den > 0.0
        self.v[den == 0] = 0.0
        self.v[non_zeros] /= den[non_zeros]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.0  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _specific_update_V(self):
        self.v *= self._update_membership(self.subs_nz, 2)

        if not self.assortative:
            UW = np.einsum("jq,aqk->ajk", self.u, self.w)
        else:
            UW = np.einsum("jk,ak->ajk", self.u, self.w)
        UWL = np.einsum("aij,ajk->aijk", self.den_updates, UW)
        den = np.einsum("aijk,aij->ik", UWL, 1.0 / self.Z)

        non_zeros = den > 0.0
        self.v[den == 0] = 0.0
        self.v[non_zeros] /= den[non_zeros]

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

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum("ik,aji->aijk", self.u, self.den_updates)
        num = np.einsum("jq,aijk->aijkq", self.v, UL)
        den = np.einsum("aijkq,aij->akq", num, 1.0 / self.Z)

        non_zeros = den > 0.0
        self.w[den == 0] = 0.0
        self.w[non_zeros] /= den[non_zeros]

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

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum("ik,aji->aijk", self.u, self.den_updates)
        num = np.einsum("jk,aijk->aijk", self.v, UL)
        den = np.einsum("aijk,aij->ak", num, 1.0 / self.Z)

        non_zeros = den > 0.0
        self.w[den == 0] = 0.0
        self.w[non_zeros] /= den[non_zeros]

    def _update_W_approx(self) -> float:
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

        UV = np.einsum(
            "Ik,Iq->Ikq", self.u[self.subs_nz[1], :], self.v[self.subs_nz[2], :]
        )
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(
                    self.subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L
                )

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum("ik,aji->aijk", self.u, self.den_updates)
        den = np.einsum("jq,aijk->akq", self.v, UL)

        non_zeros = den > 0.0
        self.w[den == 0] = 0.0
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.0  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative_approx(self) -> float:
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
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(
                self.subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L
            )

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum("ik,aji->aijk", self.u, self.den_updates)
        den = np.einsum("jk,aijk->ak", self.v, UL)

        non_zeros = den > 0.0
        self.w[den == 0] = 0.0
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.0  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_eta_approx(self) -> float:
        """
        Update pair interaction coefficient eta by using an approximation.

        Returns
        -------
        dist_eta : float
                   Maximum distance between the old and the new pair interaction coefficient eta.
        """

        den = self.lambdalambdaT.sum()
        if not den > 0.0:
            log_and_raise_error(ValueError, "eta update_approx has zero denominator!")

        self.eta = self.AAtSum / den

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.0  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)  # type: ignore
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
            log_and_raise_error(ValueError, "eta fix point has zero denominator!")
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
            self.eta = 0.0  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)  # type: ignore
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
            uttkrp_DK = sp_uttkrp_assortative(
                self.data_M_nz, subs_nz, m, self.u, self.v, self.w
            )

        return uttkrp_DK

    def _likelihood(self) -> float:
        """
        Compute the log-likelihood of the data.


        Returns
        -------
        loglik : float
            Log-likelihood value.
        """

        self.lambdalambdaT = np.einsum(
            "aij,aji->aij", self.lambda_aij, self.lambda_aij
        )  # to use in Z and eta
        self.Z = self._calculate_Z()

        ft = (self.data.data * np.log(self.lambda_nz)).sum()

        st = 0.5 * np.log(self.eta) * self.AAtSum

        tt = 0.5 * np.log(self.Z).sum()

        loglik = ft + st - tt

        if np.isnan(loglik):
            log_and_raise_error(ValueError, "log-likelihood is NaN!")

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
