"""
Class definition of CRep, the algorithm to perform inference in networks with reciprocity.
The latent variables are related to community memberships and reciprocity value.
"""

import logging
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq, root
from sktensor import dtensor, sptensor
import sktensor as skt
from typing_extensions import Unpack

from ..input.preprocessing import preprocess
from ..input.tools import (
    get_item_array_from_subs, log_and_raise_error, sp_uttkrp, sp_uttkrp_assortative)
from ..output.evaluate import func_lagrange_multiplier, lambda_full, u_with_lagrange_multiplier
from .base import ModelBase, ModelFitParameters, ModelUpdateMixin
from .constants import EPS_


class DynCRep(ModelBase, ModelUpdateMixin):
    """
    Class definition of CRepDyn_w_temp, the algorithm to perform inference in temporal  networks
    with reciprocity.
    """

    def __init__(
        self,
        inf=10000000000.0,
        err_max=0.000000000001,
        err=0.01,
        num_realizations=1,
        convergence_tol=0.0001,
        decision=10,
        max_iter=1000,
        plot_loglik=False,
        flag_conv="log",
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
            flag_conv,
        )

        # Initialize the attributes
        self.u_f = None
        self.v_f = None
        self.w_f = None

    def check_fit_params(
        self,
        K: int,
        data: Union[skt.sptensor, skt.dtensor],
        undirected: bool,
        initialization: int,
        assortative: bool,
        constrained: bool,
        constraintU: bool,
        eta0: Union[float, None],
        beta0: float,
        ag: float,
        bg: float,
        flag_data_T: int,
        **extra_params: Unpack[ModelFitParameters],
    ) -> None:
        message = (
            "The initialization parameter can be either 0, or 1.  It is used as an "
            "indicator to initialize the membership matrices u and v and the affinity  "
            "matrix w. If it is 0, they will be generated randomly, otherwise they will  "
            "upload from file."
        )  # TODO: Update this message
        available_extra_params = [
            "fix_eta",
            "fix_beta",
            "fix_w",
            "fix_communities",
            "files",
            "out_inference",
            "out_folder",
            "end_file",
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
            beta0=beta0,
            eta0=eta0,
            message=message,
            **extra_params,
        )

        self.assortative = assortative  # if True, the network is assortative
        self.constrained = constrained  # if True, use the configuration with constraints on the updates
        self.constraintU = constraintU  # if True, use constraint on U
        self.ag = ag  # shape of gamma prior
        self.bg = bg  # rate of gamma prior

        self.beta0 = beta0
        if flag_data_T not in [0, 1]:
            log_and_raise_error(ValueError, "flag_data_T has to be either 0 or 1!")
        else:
            self.flag_data_T = (
                flag_data_T  # if 0: previous time step, 1: same time step
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

        if (
            self.fix_eta
        ):  # TODO: would it make sense to define this case only if self.eta0 is not
            # None? Otherwise, mypy raises an error, giving that it leads to self.eta_old = None,
            # but somewhere there's a difference between self.eta and self.eta_old (float - None).
            self.eta = self.eta_old = self.eta_f = self.eta0  # type: ignore

        if self.fix_beta:
            self.beta = self.beta_old = self.beta_f = self.beta0  # type: ignore

        # Parameters for the initialization of the model
        self.use_unit_uniform = True
        self.normalize_rows = True

        if self.initialization > 0:
            self.theta = np.load(Path(self.files).resolve(), allow_pickle=True)

    def fit(
        self,
        data: Union[np.ndarray, skt.sptensor],
        T: int,
        nodes: List[int],
        mask: Optional[np.ndarray] = None,
        K: int = 2,
        rseed: int = 0,
        ag: float = 1.0,
        bg: float = 0.5,
        flag_data_T: int = 0,
        temporal: bool = True,
        **extra_params: Unpack[ModelFitParameters],
    ) -> Tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        float,
        np.ndarray[Any, np.dtype[np.float64]],
        float,
    ]:
        """
        Model directed networks by using a probabilistic generative model that assumes community parameters and
        reciprocity coefficient. The inference is performed via the EM algorithm.

        Parameters
        ----------
        data : ndarray or sptensor
               Graph adjacency tensor.
        T : int
            Number of time steps.
        nodes : list of int
                List of node IDs.
        mask : ndarray, optional
               Mask for selecting the held-out set in the adjacency tensor in case of cross-validation.
        K : int, default 2
            Number of communities.
        rseed : int, default 0
                Random seed.
        ag : float, default 1.0
             Shape of gamma prior.
        bg : float, default 0.5
             Rate of gamma prior.
        flag_data_T : int, default 0
                      Flag to determine which log-likelihood function to use.
        temporal : bool, default True
                   Flag to determine if the function should behave in a temporal manner.

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
        # Check the parameters for fitting the model
        self.check_fit_params(
            K,  # type: ignore
            data,
            ag=ag,
            bg=bg,
            flag_data_T=flag_data_T,
            **extra_params,
        )  # TODO: fix missing positional arguments after
        # change in extra_params

        # Set the random seed
        logging.debug("Fixing random seed to: %s", rseed)
        self.rng = np.random.RandomState(rseed)

        # Initialize the fit parameters
        self.nodes = nodes
        maxL = -self.inf
        T = max(0, min(T, data.shape[0] - 1))
        self.T = T
        self.temporal = temporal

        # Preprocess the data for fitting the model
        T, data, data_AtAtm1, data_T, data_T_vals, data_Tm1, subs_nzp = (
            self._preprocess_data_for_fit(T, data)
        )

        # Set the preprocessed data and other related variables as attributes of the class instance
        self.data_AtAtm1 = data_AtAtm1
        self.data_Tm1 = data_Tm1
        self.data_T_vals = data_T_vals
        self.subs_nzp = subs_nzp
        self.data = data
        self.data_T = data_T
        self.T = T
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
                best_loglik_values = list(loglik_values)
                self.maxL = loglik
                self.final_it = it
                conv = convergence
                self.best_r = r

            # Log the current realization number, log-likelihood, number of iterations, and elapsed time
            logging.debug(
                "Nreal = %s - Log-likelihood = %s - iterations = %s - "
                "time = %s seconds",
                r,
                loglik,
                self.final_it,
                np.round(time.time() - self.time_start, 2),
            )

        # end cycle over realizations

        # Evaluate the results of the fitting process
        self._evaluate_fit_results(self.maxL, conv, best_loglik_values)

        # Return the final parameters and the maximum log-likelihood
        return self.u_f, self.v_f, self.w_f, self.eta_f, self.beta_f, self.maxL  # type: ignore

    def _initialize_realization(self):
        """
        This method initializes the parameters for each realization of the EM algorithm.
        It also sets up local variables for convergence checking.
        """
        # Log the current state of the random number generator
        logging.debug("Random number generator seed: %s", self.rng.get_state()[1][0])

        # Call the _initialize method from the parent class to initialize the parameters for the current realization
        super()._initialize()

        # Call the _update_old_variables method from the parent class to update the old variables for the current realization
        super()._update_old_variables()

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
        self, T: int, data: Union[skt.dtensor, skt.sptensor]
    ) -> Tuple[
        int,
        Union[skt.dtensor, skt.sptensor],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple,
    ]:
        """
        Preprocess the data for fitting the model.

        Parameters
        ----------
        T : int
            Number of time steps.
        data : skt.dtensor or skt.sptensor
            The input data tensor.
        temporal : bool
            Flag to determine if the function should behave in a temporal manner.

        Returns
        -------
        tuple
            A tuple containing the number of time steps, the preprocessed data tensor, the preprocessed data tensor for
            numerator containing Aij(t)*(1-Aij(t-1)), the transposed data tensor, the values of the non-zero entries in
            the transposed data tensor, the transposed data tensor at time t-1, and the indices of the non-zero entries
            in the data tensor.
        """

        # Set the number of time steps based on whether the model is temporal or static
        self.L = T + 1 if self.temporal else 1
        logging.debug("Number of time steps L: %s", self.L)

        # Limit the data to the number of time steps
        data = data[: T + 1, :, :]

        # Initialize the data for calculating the numerator containing Aij(t)*(1-Aij(t-1))
        data_AtAtm1 = np.zeros(data.shape)
        data_AtAtm1[0, :, :] = data[0, :, :]

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
            # Copy the data at time t to the array data_Tm1
            data_Tm1 = data_T.copy()
        if self.flag_data_T == 0:
            # Create data at time t-1 as an array of zeros
            data_Tm1 = np.zeros_like(data)
            # Copy the data at time t-1 to the array
            for i in range(T):
                data_Tm1[i + 1, :, :] = data_T[i, :, :]
        self.sum_datatm1 = data_Tm1[1:].sum()

        # Calculate Aij(t)*Aij(t-1) and (1-Aij(t))*Aij(t-1)
        if T > 0:
            logging.debug(
                "T is greater than 0. Proceeding with calculations that require "
                "multiple time steps."
            )
            # Calculate Aij(t)*Aij(t-1) and (1-Aij(t))*Aij(t-1)
            bAtAtm1_l = 0
            Atm11At_l = 0
            for i in range(T):
                data_AtAtm1[i + 1, :, :] = data[i + 1, :, :] * (1 - data[i, :, :])
                # Calculate the expression Aij(t)*Aij(t-1)
                sub_nz_and = np.logical_and(data[i + 1, :, :] > 0, data[i, :, :] > 0)
                bAtAtm1_l += (
                    (data[i + 1, :, :][sub_nz_and] * data[i, :, :][sub_nz_and])
                ).sum()
                # Calculate the expression (1-Aij(t))*Aij(t-1)
                sub_nz_and = np.logical_and(
                    data[i, :, :] > 0, (1 - data[i + 1, :, :]) > 0
                )
                Atm11At_l += (
                    ((1 - data[i + 1, :, :])[sub_nz_and] * data[i, :, :][sub_nz_and])
                ).sum()
            self.bAtAtm1 = bAtAtm1_l
            self.Atm11At = Atm11At_l

        # Calculate the sum of the data hat from 1 to T
        self.sum_data_hat = data_AtAtm1[1:].sum()

        # Calculate the denominator containing Aji(t)
        data_T_vals = get_item_array_from_subs(data_Tm1, data_AtAtm1.nonzero())

        # Preprocess the data to handle the sparsity
        data_AtAtm1 = preprocess(data_AtAtm1)
        data = preprocess(data)

        # Save the indices of the non-zero entries
        subs_nzp = (
            data_AtAtm1.nonzero()
            if isinstance(data_AtAtm1, skt.dtensor)
            else data_AtAtm1.subs
        )

        # Initialize the beta hat
        self.beta_hat = np.ones(T + 1)
        if T > 0:
            self.beta_hat[1:] = self.beta0

        return T, data, data_AtAtm1, data_T, data_T_vals, data_Tm1, subs_nzp

    def compute_likelihood(self):
        """
        Compute the likelihood of the model.
        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        data_T : sptensor/dtensor
                 Transposed graph adjacency tensor.
        data_T_vals : ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : tuple
                    Indices of elements of data that are non-zero.
        T : int
            Number of time steps.
        mask : ndarray, optional
               Mask for selecting the held-out set in the adjacency tensor in case of cross-validation.
        Returns
        -------
        loglik : float
                 Log-likelihood of the model.

        """
        return self._Likelihood(
            self.data_AtAtm1,
            self.data_Tm1,
            self.data_T_vals,
            self.subs_nzp,
            self.T,
            self.mask,
        )

    def _initialize_beta(self) -> None:

        # If beta0 is not None, assign its value to beta
        if self.beta0 is not None:
            self.beta = self.beta0

    def _file_initialization(self) -> None:
        """
        Initialize u, v, w_dyn or w_stat using the input file.
        """
        # Call the base class methods to initialize u and v
        self._initialize_u()
        self._initialize_v()
        # If temporal is True, initialize w dynamically
        if self.temporal:
            self._initialize_w_dyn()
        else:
            # If temporal is False, initialize w statically
            self._initialize_w_stat()

    def _random_initialization(self) -> None:
        # Call the random initialization method from the parent class
        super()._random_initialization()
        # Randomize beta
        self._randomize_beta(1)  # Generates a single random number

    def _update_cache(
        self,
        data: Union[dtensor, sptensor],
        data_T_vals: np.ndarray,
        subs_nz: Tuple[np.ndarray],
    ) -> None:
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
        self.lambda0_nz = super()._lambda_nz(subs_nz, temporal=self.temporal)
        self.M_nz = self.lambda0_nz + self.eta * data_T_vals  # [np.newaxis,:]
        self.M_nz[self.M_nz == 0] = 1

        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
            self.data_rho2 = (
                (data[subs_nz] * self.eta * data_T_vals) / self.M_nz
            ).sum()
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz
            self.data_rho2 = ((data.vals * self.eta * data_T_vals) / self.M_nz).sum()

    def _update_em(self) -> Tuple[float, float, float, float, float]:
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
        self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)

        if not self.fix_communities:
            d_u = self._update_U(self.subs_nzp)
            self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)
            if self.undirected:
                self.v = self.u
                self.v_old = self.v
                d_v = d_u
            else:
                d_v = self._update_V(self.subs_nzp)
                self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)
        else:
            d_u = 0
            d_v = 0

        if not self.fix_w:
            if not self.assortative:
                if self.temporal:
                    d_w = self._update_W_dyn(self.subs_nzp)
                else:
                    d_w = self._update_W_stat(self.subs_nzp)
            else:
                if self.temporal:
                    d_w = self._update_W_assortative_dyn(self.subs_nzp)
                else:
                    d_w = self._update_W_assortative_stat(self.subs_nzp)
            self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)
        else:
            d_w = 0

        if not self.fix_beta:
            if self.T > 0:
                d_beta = self._update_beta()
                self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)
            else:
                d_beta = 0.0
        else:
            d_beta = 0.0

        if not self.fix_eta:
            denominator = self.E0 + self.Etg0 * self.beta_hat[-1]
            d_eta = self._update_eta(denominator=denominator)
        else:
            d_eta = 0.0
        self._update_cache(self.data_AtAtm1, self.data_T_vals, self.subs_nzp)

        return d_u, d_v, d_w, d_eta, d_beta

    def _update_eta(self, denominator: float) -> float:
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

        if self.eta < 0 or self.eta > 1:

            message = f"Eta has to be a positive number! Current value is {self.eta}"
            log_and_raise_error(ValueError, message)

        dist_eta = abs(self.eta - self.eta_old)  # type: ignore
        self.eta_old = float(self.eta)

        return dist_eta

    def _update_beta(self):
        """
        Update beta.
        Returns
        -------
        dist_beta : float
                    Maximum distance between the old and the new beta.
        """
        self.beta = brentq(self.func_beta_static, a=0.001, b=0.999)
        self.beta_hat[1:] = self.beta

        dist_beta = abs(self.beta - self.beta_old)
        self.beta_old = np.copy(self.beta)

        return dist_beta

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

        # Call the specific update U method
        self._specific_update_U(subs_nz)

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _specific_update_U(self, subs_nz):
        """
        Specific logic for updating U in the DynCRep class.
        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        if self.constraintU:
            u_tmp = self.u_old * (
                self._update_membership(subs_nz, self.u, self.v, self.w, 1)
            )

            Du = np.einsum("iq->q", self.v)
            if not self.assortative:
                w_k = np.einsum("a,akq->kq", self.beta_hat, self.w)
                Z_uk = np.einsum("q,kq->k", Du, w_k)
            else:
                w_k = np.einsum("a,ak->k", self.beta_hat, self.w)
                Z_uk = np.einsum("k,k->k", Du, w_k)

            for i in range(self.u.shape[0]):
                lambda_i = self.enforce_constraintU(u_tmp[i], Z_uk)
                self.u[i] = abs(u_tmp[i] / (lambda_i + Z_uk))

        else:

            self.u = (self.ag - 1) + self.u_old * (
                self._update_membership(subs_nz, self.u, self.v, self.w, 1)
            )

            if not self.constrained:
                Du = np.einsum("iq->q", self.v)
                if not self.assortative:
                    w_k = np.einsum("a,akq->kq", self.beta_hat, self.w)
                    Z_uk = np.einsum("q,kq->k", Du, w_k)
                else:
                    w_k = np.einsum("a,ak->k", self.beta_hat, self.w)
                    Z_uk = np.einsum("k,k->k", Du, w_k)
                non_zeros = Z_uk > 0.0
                self.u[:, Z_uk == 0] = 0.0
                self.u[:, non_zeros] /= self.bg + Z_uk[np.newaxis, non_zeros]
            else:
                Du = np.einsum("iq->q", self.v)
                if not self.assortative:
                    w_k = np.einsum("a,akq->kq", self.beta_hat, self.w)
                    Z_uk = np.einsum("q,kq->k", Du, w_k)
                else:
                    w_k = np.einsum("a,ak->k", self.beta_hat, self.w)
                    Z_uk = np.einsum("k,k->k", Du, w_k)
                for i in range(self.u.shape[0]):
                    if self.u[i].sum() > self.err_max:
                        u_root = root(
                            u_with_lagrange_multiplier,
                            self.u_old[i],
                            args=(self.u[i], Z_uk),
                        )
                        self.u[i] = u_root.x

    def _update_V(self, subs_nz):
        """
        Update in-coming membership matrix.
        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        Returns
        -------
        dist_v : float
                 Maximum distance between the old and the new membership matrix v.
        """

        # Call the specific update V method
        self._specific_update_V(subs_nz)

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _specific_update_V(self, subs_nz):
        """
        Specific logic for updating V in the DynCRep class.
        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.
        """

        self.v = (self.ag - 1) + self.v_old * self._update_membership(
            subs_nz, self.u, self.v, self.w, 2
        )

        if not self.constrained:
            Dv = np.einsum("iq->q", self.u)

            if not self.assortative:
                w_k = np.einsum("a,aqk->qk", self.beta_hat, self.w)
                Z_vk = np.einsum("q,qk->k", Dv, w_k)
            else:
                w_k = np.einsum("a,ak->k", self.beta_hat, self.w)
                Z_vk = np.einsum("k,k->k", Dv, w_k)
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.0
            self.v[:, non_zeros] /= self.bg + Z_vk[np.newaxis, non_zeros]
        else:
            Dv = np.einsum("iq->q", self.u)
            if not self.assortative:
                w_k = np.einsum("a,aqk->qk", self.beta_hat, self.w)
                Z_vk = np.einsum("q,qk->k", Dv, w_k)
            else:
                w_k = np.einsum("a,ak->k", self.beta_hat, self.w)
                Z_vk = np.einsum("k,k->k", Dv, w_k)

            for i in range(self.v.shape[0]):
                if self.v[i].sum() > self.err_max:
                    v_root = root(
                        u_with_lagrange_multiplier,
                        self.v_old[i],
                        args=(self.v[i], Z_vk),
                    )
                    self.v[i] = v_root.x

    def _specific_update_W_dyn(self, subs_nz: tuple):

        uttkrp_DKQ = np.zeros_like(self.w)

        for idx, (a, i, j) in enumerate(zip(*subs_nz)):
            uttkrp_DKQ[a, :, :] += self.data_M_nz[idx] * np.einsum(
                "k,q->kq", self.u[i], self.v[j]
            )

        self.w = self.w * uttkrp_DKQ

        Z = np.einsum("k,q->kq", self.u.sum(axis=0), self.v.sum(axis=0))
        Z = np.einsum("a,kq->akq", self.beta_hat, Z)

        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

    def _update_W_dyn(self, subs_nz):
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

        self._specific_update_W_dyn(subs_nz)

        # TODO: Ask why this is commented out
        # low_values_indices = self.w < self.err_max  # values are too low
        # self.w[low_values_indices] = 0. #self.err_max  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w_old)

        return dist_w

    def _specific_update_W_stat(self, subs_nz: tuple):

        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum("Ik,Iq->Ikq", self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV

        for _a, k, q in zip(*sub_w_nz):
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

    def _update_W_stat(self, subs_nz):
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

        self._specific_update_W_stat(subs_nz)

        dist_w, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

        return dist_w

    def _specific_update_W_assortative_dyn(self, subs_nz: tuple):
        uttkrp_DKQ = np.zeros_like(self.w)

        for idx, (a, i, j) in enumerate(zip(*subs_nz)):
            uttkrp_DKQ[a, :] += self.data_M_nz[idx] * self.u[i] * self.v[j]

        self.w = (self.ag - 1) + self.w * uttkrp_DKQ

        Z = (self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0))
        Z = np.einsum("a,k->ak", self.beta_hat, Z)
        Z += self.bg

        non_zeros = Z > 0

        self.w[non_zeros] /= Z[non_zeros]

    def _update_W_assortative_dyn(self, subs_nz):
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

        self._specific_update_W_assortative_dyn(subs_nz)

        dist_w, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

        return dist_w

    def _specific_update_W_assortative_stat(self, subs_nz: tuple):
        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum("Ik,Ik->Ik", self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV

        for _, k in zip(*sub_w_nz):
            uttkrp_DKQ[:, k] += np.bincount(
                subs_nz[0], weights=uttkrp_I[:, k], minlength=1
            )[0]

        self.w = (self.ag - 1) + self.w * uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
        Z *= 1.0 + self.beta_hat[self.T] * self.T
        Z += self.bg

        non_zeros = Z > 0
        self.w[non_zeros] /= Z[non_zeros]

    def _update_W_assortative_stat(self, subs_nz):
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

        self._specific_update_W_assortative_stat(subs_nz)

        dist_w, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

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
            uttkrp_DK = sp_uttkrp(
                self.data_M_nz, subs_nz, m, u, v, w, temporal=self.temporal
            )
        else:
            uttkrp_DK = sp_uttkrp_assortative(
                self.data_M_nz, subs_nz, m, u, v, w, temporal=self.temporal
            )
        return uttkrp_DK

    def _Likelihood(
        self,
        data: Union[dtensor, sptensor],
        data_T: Union[dtensor, sptensor],
        data_T_vals: np.ndarray,
        subs_nz: Tuple[np.ndarray],
        T: int,
        mask: Optional[np.ndarray] = None,
        EPS: float = EPS_,
    ) -> float:
        """
        Compute the pseudo log-likelihood of the data.

        Parameters
        ----------
        data : Union[dtensor, sptensor]
               Graph adjacency tensor.
        data_T : Union[dtensor, sptensor]
                 Graph adjacency tensor (transpose).
        data_T_vals : np.ndarray
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : Tuple[np.ndarray]
                  Indices of elements of data that are non-zero.
        T : int
            Number of time steps.
        mask : Optional[np.ndarray]
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        EPS : float, default 1e-12
              Small constant to prevent division by zero.

        Returns
        -------
        l : float
            Pseudo log-likelihood value.
        """

        self._update_cache(data, data_T_vals, subs_nz)

        if not self.assortative:
            w_k = np.einsum("a,akq->akq", self.beta_hat, self.w)
        else:
            w_k = np.einsum("a,ak->ak", self.beta_hat, self.w)

        lambda0_ija_loc = lambda_full(self.u, self.v, w_k)

        if mask is not None:
            sub_mask_nz = mask.nonzero()
            if isinstance(data, skt.dtensor):
                loglik = (
                    -(1 + self.beta0) * self.lambda0_ija[sub_mask_nz].sum()
                    - self.eta
                    * (data_T[sub_mask_nz] * self.beta_hat[sub_mask_nz[0]]).sum()
                )
            elif isinstance(data, skt.sptensor):
                loglik = (
                    -(1 + self.beta0) * self.lambda0_ija[sub_mask_nz].sum()
                    - self.eta
                    * (
                        data_T.toarray()[sub_mask_nz] * self.beta_hat[sub_mask_nz[0]]
                    ).sum()
                )
        else:
            if isinstance(data, skt.dtensor):
                loglik = -(1 + self.beta0) * self.lambda0_ija.sum() - self.eta * (
                    data_T[0].sum() + self.beta0 * data_T[1:].sum()
                )
            elif isinstance(data, skt.sptensor):
                loglik = (
                    -lambda0_ija_loc.sum()
                    - self.eta * (data_T.sum(axis=(1, 2)) * self.beta_hat).sum()
                )

        logM = np.log(self.M_nz)
        if isinstance(data, skt.dtensor):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, skt.sptensor):
            Alog = (data.vals * logM).sum()
        loglik += Alog

        loglik += (np.log(self.beta_hat[subs_nz[0]] + EPS) * data.vals).sum()
        if self.T > 0:
            loglik += (np.log(1 - self.beta_hat[-1] + EPS) * self.bAtAtm1).sum()
            loglik += (np.log(self.beta_hat[-1] + EPS) * self.Atm11At).sum()

        if not self.constraintU:
            if self.ag >= 1.0:
                loglik += (self.ag - 1) * np.log(self.u + EPS).sum()
                loglik += (self.ag - 1) * np.log(self.v + EPS).sum()
            if self.bg >= 0.0:
                loglik -= self.bg * self.u.sum()
                loglik -= self.bg * self.v.sum()

        if np.isnan(loglik):
            message = "Likelihood is NaN!"
            log_and_raise_error(ValueError, message)

        return loglik

    def enforce_constraintU(self, num: float, den: float) -> float:
        """
        Enforce a constraint on the U matrix during the model fitting process.
        It uses the root finding algorithm to find the value of lambda that satisfies the constraint.

        Parameters
        ----------
        num : float
            The numerator part of the constraint equation.
        den : float
            The denominator part of the constraint equation.

        Returns
        -------
        lambda_i : float
            The value of lambda that satisfies the constraint.
        """
        lambda_i_test = root(func_lagrange_multiplier, 0.1, args=(num, den))
        lambda_i = lambda_i_test.x

        return lambda_i

    def func_beta_static(self, beta_t: float) -> float:
        """
        Calculate the value of beta at time t for the static model.

        Parameters
        ----------
        beta_t : float
            The value of beta at time t.

        Returns
        -------
        bt : float
            The calculated value of beta at time t for the static model.
        """

        if self.assortative:
            lambda0_ija = np.einsum(
                "k,k->k", self.u.sum(axis=0), self.w[1:].sum(axis=0)
            )
        else:
            lambda0_ija = np.einsum(
                "k,kq->q", self.u.sum(axis=0), self.w[1:].sum(axis=0)
            )
        lambda0_ija = np.einsum("k,k->", self.v.sum(axis=0), lambda0_ija)

        bt = -(lambda0_ija + self.eta * self.sum_datatm1)
        bt -= self.bAtAtm1 / (1 - beta_t)  # adding Aij(t-1)*Aij(t)

        bt += self.sum_data_hat / beta_t  # adding sum A_hat from 1 to T
        bt += self.Atm11At / beta_t  # adding Aij(t-1)*(1-Aij(t))
        return bt

    def _copy_variables(
        self, source_suffix: str, target_suffix: str
    ) -> None:
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

        source_var = getattr(self, f"beta{source_suffix}")
        setattr(self, f"beta{target_suffix}", np.copy(source_var))
