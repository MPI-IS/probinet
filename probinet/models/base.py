"""
Base classes for the models classes.
"""

import dataclasses
import logging
import os
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sparse import COO

from probinet.input.loader import build_adjacency_from_file
from probinet.models.classes import GraphData
from probinet.models.constants import CONVERGENCE_TOL_, DECISION_, ERR_, ERR_MAX_, INF_
from probinet.types import GraphDataType
from probinet.utils.tools import log_and_raise_error
from probinet.visualization.plot import plot_L


@dataclasses.dataclass
class ModelBaseParameters:
    """
    Attributes
    ----------
    inf : float
        Initial value of the log-likelihood.
    err_max : float
        Minimum value for the parameters.
    err : float
        Noise for the initialization.
    num_realizations : int
        Number of iterations with different random initialization.
    convergence_tol : float
        Tolerance for convergence.
    decision : int
        Convergence parameter.
    max_iter : int
        Maximum number of EM steps before aborting.
    plot_loglik : bool
        Flag to plot the log-likelihood.
    flag_conv : str
        Flag to choose the convergence criterion.
    """

    inf: float = INF_  # initial value of the log-likelihood
    err_max: float = ERR_MAX_  # minimum value for the parameters
    err: float = ERR_  # noise for the initialization
    num_realizations: int = (
        3  # number of iterations with different random initialization
    )
    convergence_tol: float = CONVERGENCE_TOL_  # tolerance for convergence
    decision: int = DECISION_  # convergence parameter
    max_iter: int = 500  # maximum number of EM steps before aborting
    plot_loglik: bool = False  # flag to plot the log-likelihood
    flag_conv: str = "log"  # flag to choose the convergence criterion


class ModelBase(ModelBaseParameters):
    """
    Base class for the models classes that inherit from the ModelBaseParameters class. It contains the
    methods to check the parameters of the fit method, initialize the parameters, and check for
    convergence. All the models classes should inherit from this class.
    """

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class
        super().__init__(*args, **kwargs)
        self.__doc__ = ModelBaseParameters.__doc__
        # Define additional attributes
        self.attributes_to_save_names = [
            "u_f",
            "v_f",
            "w_f",
            "eta_f",
            "final_it",
            "maxL",
            "maxPSL",
            "beta_f",
            "nodes",
            "pibr",
            "mupr",
        ]

        # Initialize the attributes
        self.u_f: np.ndarray = np.array([])
        self.v_f: np.ndarray = np.array([])
        self.w_f: np.ndarray = np.array([])
        self.use_unit_uniform = False
        self.theta: Dict[str, Any] = {}
        self.normalize_rows = False
        self.nodes: list[Any] = []
        self.rng = np.random.default_rng()
        self.beta_hat: np.ndarray = np.array([])
        self.best_r: int = 0
        self.final_it: int = 0

        self.message_for_invalid_initialization = (
            "The initialization parameter can be either 0 "
            "or 1. If 0, the model will be initialized "
            "randomly. If 1, the model will be initialized with the parameters "
            "stored in the file specified in the `files` parameter."
        )

    def check_params_to_load_data(self, binary, noselfloop, undirected, **kwargs):
        """
        Check that the parameters to load the data are correct.
        """
        pass

    def _validate_eta0(self, eta0: float) -> None:
        if eta0 is not None and eta0 <= 0.0:
            message = "If not None, the eta0 parameter has to be greater than 0.!"
            log_and_raise_error(ValueError, message)

    def _validate_undirected_eta(self) -> None:
        if self.undirected and not (self.fix_eta and self.eta0 == 1):
            message = (
                "If undirected=True, the parameter eta has to be fixed equal to 1 "
                "(s.t. log(eta)=0)."
            )
            log_and_raise_error(ValueError, message)

    def _check_fit_params(self, *args, **kwargs) -> None:
        """
        Check the parameters of the fit method.
        """
        # Extract parameters from args and kwargs
        initialization = kwargs.get(
            "initialization", args[0] if len(args) > 0 else None
        )
        undirected = kwargs.get("undirected", args[1] if len(args) > 1 else None)
        assortative = kwargs.get("assortative", args[2] if len(args) > 2 else None)
        data = kwargs.get("data", args[3] if len(args) > 3 else None)
        K = kwargs.get("K", args[4] if len(args) > 4 else None)
        data_X = kwargs.get("data_X", args[5] if len(args) > 5 else None)
        eta0 = kwargs.get("eta0", args[6] if len(args) > 6 else None)
        beta0 = kwargs.get("beta0", args[7] if len(args) > 7 else None)
        gamma = kwargs.get("gamma", args[8] if len(args) > 8 else None)
        use_approximation = kwargs.get("use_approximation", False)
        temporal = kwargs.get("temporal", True)
        fix_eta = kwargs.get("fix_eta", False)
        fix_w = kwargs.get("fix_w", False)
        fix_communities = kwargs.get("fix_communities", False)
        fix_beta = kwargs.get("fix_beta", None)
        files = kwargs.get("files", None)
        fix_pibr = kwargs.get("fix_pibr", None)
        fix_mupr = kwargs.get("fix_mupr", None)
        out_inference = kwargs.get("out_inference", False)
        out_folder = kwargs.get("out_folder", Path("outputs"))
        end_file = kwargs.get("end_file", " ")
        verbose = kwargs.get("verbose", 0)

        # Check the initialization parameter
        logging.debug("Initialization parameter: %s", initialization)
        if initialization not in {0, 1}:
            log_and_raise_error(ValueError, self.message_for_invalid_initialization)
        self.initialization = initialization

        if initialization == 1:
            if files is None:
                log_and_raise_error(
                    ValueError, "If initialization is 1, provide a file."
                )
            self.files = files

        self.undirected = undirected
        self.assortative = assortative
        self.use_approximation = use_approximation
        self.temporal = temporal

        self.fix_eta = fix_eta
        self.fix_beta = fix_beta
        self.fix_w = fix_w
        self.fix_communities = fix_communities
        self.fix_pibr = fix_pibr
        self.fix_mupr = fix_mupr

        self.out_inference = out_inference
        self.out_folder = out_folder
        self.end_file = end_file
        self.verbose = verbose

        self.N = data.shape[1]
        self.L = data.shape[0]
        self.K = K

        # Validate the data_X parameter
        if data_X is not None:
            logging.debug("Data_X parameter: %s", data_X.shape)
            if data_X.shape[0] != self.N:
                message = "The number of rows of the data_X matrix is different from the number of nodes."
                log_and_raise_error(ValueError, message)

        # Validate the eta0 parameter
        logging.debug("Eta0 parameter: %s", eta0)
        if self.fix_eta and self.eta0 is None:
            log_and_raise_error(
                ValueError, "If fix_eta=True, provide a value for eta0."
            )

        if self.fix_beta:
            if beta0 is None:
                log_and_raise_error(
                    ValueError, "If fix_beta=True, provide a value for beta0."
                )
            else:
                self.beta0 = beta0

        if self.fix_w:
            logging.debug("Fixing w.")
            if self.initialization != 1:
                message = "If fix_w=True, the initialization has to be  1."
                log_and_raise_error(ValueError, message)

        if self.fix_communities:
            if self.initialization != 1:
                message = "If fix_communities=True, the initialization has to be 1."
                log_and_raise_error(ValueError, message)

    def _initialize(self) -> None:
        """
        Initialization of the parameters u, v, w, eta.
        """
        # Call the method to initialize eta
        self._initialize_eta()

        # Call the method to initialize beta
        self._initialize_beta()

        # Check the initialization type and call the corresponding method
        if self.initialization == 0:
            # If initialization type is 0, call the method for random initialization
            self._random_initialization()
        elif self.initialization == 1:
            # If initialization type is 1, call the method for file-based initialization
            self._file_initialization()
        else:
            # If initialization type is neither 0 nor 1, log a message and raise a ValueError
            message = "Invalid initialization parameter."
            log_and_raise_error(ValueError, message)

    def _initialize_eta(self) -> None:
        # If eta0 is not None, assign its value to eta
        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            # If eta0 is None, log a message and call the method to randomize eta
            logging.debug("eta is initialized randomly.")
            self._randomize_eta(use_unit_uniform=self.use_unit_uniform)

    @abstractmethod
    def _initialize_beta(self) -> None:
        """
        Placeholder function for initializing beta.
        Intentionally left empty for subclasses to override if necessary.
        """

    def _initialize_beta_from_file(self) -> None:
        # Assign the beta matrix from the input file to the beta attribute
        self.beta = self.theta["beta"]

        # Add random noise to the beta matrix
        self.beta = self._add_random_noise(self.beta)

    def _random_initialization(self) -> None:
        # Log a message indicating that u, v and w are being initialized randomly
        logging.debug("%s", "u, v and w are initialized randomly.")

        # Randomize w and u, v
        self._randomize_w()
        self._randomize_u_v(normalize_rows=self.normalize_rows)

    def _file_initialization(self) -> None:
        # Log a message indicating that u, v and w are being initialized using the input file
        logging.debug("u, v and w are initialized using the input file: %s", self.files)
        # Initialize u and v
        self._initialize_u()
        self._initialize_v()
        self._initialize_w()

    def _initialize_membership_matrix(
        self, matrix_name: str, matrix_value: np.ndarray
    ) -> None:
        # Assign the input matrix value to the local variable 'matrix'
        matrix = matrix_value

        # Assert that the nodes in the current object and the nodes in the theta dictionary are the same
        # If they are not the same, raise an AssertionError with the message 'Nodes do not match.'
        assert np.array_equal(self.nodes, self.theta["nodes"]), "Nodes do not match."

        # Find the maximum value in the 'matrix'
        max_entry = np.max(matrix)

        # Add random noise to the 'matrix'. The noise is a random number between 0 and 1,
        # multiplied by the maximum entry in the 'matrix' and the error rate 'self.err'
        matrix += max_entry * self.err * self.rng.random(matrix.shape)

        # Set the attribute of the current object with the name 'matrix_name' to
        # the value of 'matrix'
        setattr(self, matrix_name, matrix)

    def _initialize_u(self) -> None:
        """
        Initialize out-going membership matrix u from file.
        """
        self._initialize_membership_matrix("u", self.theta["u"])

    def _initialize_v(self) -> None:
        """
        Initialize in-coming membership matrix v from file.
        """
        if self.undirected:
            self.v = self.u
        else:
            self._initialize_membership_matrix("v", self.theta["v"])

    def _add_random_noise(self, matrix: np.ndarray) -> np.ndarray:
        """
        Add random noise to a matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to which random noise will be added.

        Returns
        -------
        matrix : np.ndarray
            The matrix after random noise has been added.
        """
        max_entry = np.max(matrix)
        matrix += max_entry * self.err * self.rng.random(matrix.shape)
        return matrix

    def _initialize_w(self) -> None:
        """
        Initialize affinity tensor w from file.
        """
        self.w = self.theta["w"]
        if self.assortative:
            assert self.w.shape == (
                self.L,
                self.K,
            ), "The shape of the affinity tensor w is incorrect."

        self.w = self._add_random_noise(self.w)

    def _initialize_w_dyn(self):
        # Initialize the affinity tensor w from the input file
        w0 = self.theta["w"]
        # Initialize the affinity tensor w with zeros
        self.w = np.zeros((self.L, self.K, self.K), dtype=float)
        if self.assortative:
            if w0.ndim == 2:
                self.w = w0[np.newaxis, :].copy()
            else:
                self.w = np.diag(w0)[np.newaxis, :].copy()
        else:
            self.w[:] = w0.copy()
        # Add random noise to the affinity tensor w if fix_w is False
        if not self.fix_w:
            self.w = self._add_random_noise(self.w)

    def _initialize_w_stat(self):
        # Initialize the affinity tensor w from the input file
        w0 = self.theta["w"]
        # Initialize the affinity tensor w with zeros
        if self.assortative:
            self.w = np.zeros((1, self.K), dtype=float)
            self.w[:] = (np.diag(w0)).copy()
        else:
            self.w = np.zeros((1, self.K, self.K), dtype=float)
            self.w[:] = w0.copy()
        # Add random noise to the affinity tensor w if fix_w is False
        if not self.fix_w:
            self.w = self._add_random_noise(self.w)

    @singledispatchmethod
    def _randomize_beta(self, shape: int) -> None:
        """
        Initialize community-parameter matrix beta from file.

        Parameters
        ----------
        shape : int
                The shape of the beta matrix.
        """
        self.beta = self.rng.random(shape)

    @_randomize_beta.register
    def _(self, shape: tuple):
        """
        Assign a random number in (0, 1.) to each entry of the beta matrix, and normalize each row.
        Parameters
        ----------
        shape : tuple
                The shape of the beta matrix.
        """
        self.beta = self.rng.random(shape)
        self.beta = (self.beta.T / np.sum(self.beta, axis=1)).T

    def _randomize_u_v(self, normalize_rows: bool = True) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the membership matrices u and v,
        and normalize each row if normalize_rows is True.

        Parameters
        ----------
        normalize_rows : bool
                         If True, normalize each row of the membership matrices u and v.
        """

        self.u = self.rng.random((self.N, self.K))
        # Normalize each row of the membership matrix u
        if normalize_rows:
            # Compute the sum of each row of the membership matrix u
            row_sums = self.u.sum(axis=1)
            # Normalize each row of the membership matrix u if the sum of the row is greater than 0
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            # Set the membership matrix v to be a random sample of shape (self.N, self.K)
            self.v = self.rng.random((self.N, self.K))
            # Normalize each row of the membership matrix v
            if normalize_rows:
                row_sums = self.v.sum(axis=1)
                self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            # If the graph is undirected, set the membership matrix v to be equal to the
            # membership matrix u
            self.v = self.u

    def _randomize_w(self) -> None:
        """
        Assign a random number in (0, 1.) to each entry of the affinity tensor w.
        """

        if self.assortative:
            self.w = self.rng.random((self.L, self.K))
        else:
            self.w = self.rng.random((self.L, self.K, self.K))

    def _randomize_eta(self, use_unit_uniform: bool = False) -> None:
        """
        Generate a random number in (0, 1.) or (1., 50.) based on the flag.
        For CRep the default is (0, 1.) and for JointCRep the default is (1., 50.).

        Parameters
        ----------
        use_unit_uniform : bool
            If True, generate a random number in (1., 50.).
            If False, generate a random number in (0, 1.).
        """

        if use_unit_uniform:
            self.eta = float((self.rng.random(1)[0]))
        else:
            self.eta = self.rng.uniform(1.01, 49.99)

    def _update_old_variables(self) -> None:
        """
        Update values of the parameters in the previous iteration.
        """
        self._copy_variables(source_suffix="", target_suffix="_old")  # type: ignore

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """
        self._copy_variables(source_suffix="", target_suffix="_f")  # type: ignore

    def _lambda_nz(self, subs_nz: tuple, temporal: bool = True) -> np.ndarray:
        """
        Compute the mean lambda_ij for only non-zero entries.

        Parameters
        ----------
        subs_nz : tuple
                  Indices of elements of data that are non-zero.

        Returns
        -------
        nz_recon_I : ndarray
                     Mean lambda_ij for only non-zero entries.
        """
        if temporal:
            if not self.assortative:
                nz_recon_IQ = np.einsum(
                    "Ik,Ikq->Iq", self.u[subs_nz[1], :], self.w[subs_nz[0], :, :]
                )
            else:
                nz_recon_IQ = np.einsum(
                    "Ik,Ik->Ik", self.u[subs_nz[1], :], self.w[subs_nz[0], :]
                )

        else:
            if not self.assortative:
                nz_recon_IQ = np.einsum(
                    "Ik,kq->Iq", self.u[subs_nz[1], :], self.w[0, :, :]
                )
            else:
                nz_recon_IQ = np.einsum("Ik,k->Ik", self.u[subs_nz[1], :], self.w[0, :])

        nz_recon_I = np.einsum("Iq,Iq->I", nz_recon_IQ, self.v[subs_nz[2], :])

        return nz_recon_I

    def _ps_likelihood(
        self,
        data: GraphDataType,
        data_T: COO,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Compute the pseudo-log-likelihood.
        """

    def _likelihood(self):
        """
        Compute the log-likelihood.
        """

    def _check_for_convergence(
        self, it: int, loglik: float, coincide: int, convergence: bool
    ) -> Tuple[int, float, int, bool]:
        """
        Check for convergence of the models.

        Parameters
        ----------
        data : GraphDataType
               Graph adjacency tensor.
        it : int
             Current iteration number.
        loglik : float
                 Current log-likelihood value.
        coincide : int
                   Number of times the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        use_pseudo_likelihood : bool, default False
                                Flag to indicate whether to use pseudo likelihood.
        data_T_vals : Optional[np.ndarray]
                      Array with values of entries A[j, i] given non-zero entry (i, j).
        subs_nz : Optional[Tuple[np.ndarray]]
                  Indices of elements of data that are non-zero.
        T : Optional[int]
            Number of time steps.
        data_T : Optional[GraphDataType]
                 Graph adjacency tensor (transpose).
        mask : Optional[np.ndarray]
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
        kwargs : Union[np.ndarray, int, List[int], Tuple[np.ndarray]]
                 Additional parameters that might be needed for the computation.

        Returns
        -------
        it : int
             Updated iteration number.
        loglik : float
                 Updated log-likelihood value.
        coincide : int
                   Updated number of times the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Updated flag for convergence.
        """

        # Check for convergence
        if it % 10 == 0:
            old_L = loglik
            loglik = self.compute_likelihood()
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        # Define the convergence criterion
        convergence = coincide > self.decision or convergence
        # Update the number of iterations
        it += 1

        return it, loglik, coincide, convergence

    def _check_for_convergence_delta(
        self,
        it: int,
        coincide: int,
        du: float,
        dv: float,
        dw: float,
        de: float,
        convergence: bool,
    ) -> Tuple[int, int, bool]:
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

        if (
            du < self.convergence_tol
            and dv < self.convergence_tol
            and dw < self.convergence_tol
            and de < self.convergence_tol
        ):
            coincide += 1
        else:
            coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _output_results(self) -> None:
        """
        Output results.

        Parameters
        ----------
        maxL : float
               Maximum log-likelihood.
        nodes : list
                List of nodes IDs.
        """
        # Check if the evaluation folder exists, otherwise create it
        output_path = Path(self.out_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        # Define the evaluation file
        end_file = self.end_file if self.end_file is not None else ""
        outfile = (Path(self.out_folder) / f"theta{end_file}").with_suffix(".npz")

        # Create a dictionary to hold the attributes to be saved
        attributes_to_save = {}

        # Iterate over the instance's attributes
        for attr_name, attr_value in self.__dict__.items():
            # Check if the attribute is a numpy array and its name is in the list
            if attr_name in self.attributes_to_save_names:
                # Remove the '_f' suffix from the attribute name if it exists
                attr_name_clean = attr_name.removesuffix("_f")
                # Remove the 'pr' or 'br' suffix if it exists
                attr_name_clean = attr_name_clean.removesuffix("pr")
                attr_name_clean = attr_name_clean.removesuffix("br")
                # Add the attribute to the dictionary with the cleaned name
                attributes_to_save[attr_name_clean] = attr_value

        # Save the attributes
        np.savez_compressed(outfile, **attributes_to_save)

        logging.info("Inferred parameters saved in: %s", outfile.resolve())
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _evaluate_fit_results(
        self, maxL: float, conv: bool, best_loglik_values: list[float]
    ) -> None:
        """
        Evaluate the results of the fitting process and log the results.

        Parameters
        ----------
        maxL : float
            The maximum log-likelihood obtained from the fitting process.
        conv : bool
            A flag indicating whether the fitting process has converged.
        best_loglik_values : list of float, optional
            A list of the best log-likelihood values obtained at each iteration of the fitting
            process.
            If not provided, it defaults to None.
        """
        if not best_loglik_values:
            if self.flag_conv == "log":
                message = "Algorithm did not converge. Please increase the number of realizations, or the maximum number of iterations."
                log_and_raise_error(RuntimeError, message)
            else:
                logging.debug(
                    "The conv_flag is 'deltas'; there is no best log-likelihood."
                )
            return

        # Log the best realization, maximum log-likelihood, and the best iterations
        logging.debug(
            "Best realization = %s - maxL = %s - best iterations = %s",
            self.best_r,
            maxL,
            self.final_it,
        )

        # Log that the algorithm did converge
        if conv:
            logging.info(
                "Algorithm successfully converged after %d iterations with a maximum log-likelihood of %.4f.",
                self.final_it,
                maxL,
            )
        else:
            logging.info("Algorithm did not converge.")

        # Check if the fitting process has converged
        if np.logical_and(self.final_it == self.max_iter, not conv):
            logging.warning(
                "Solution failed to converge in %s EM steps!", self.max_iter
            )
            logging.warning(
                "Parameters won't be saved for this realization! If you have "
                "provided a number of realizations equal to one, please increase it."
            )
            return

        # If the fitting process has converged and out_inference is True, evaluate the results
        if self.out_inference:
            self._output_results()
        else:
            logging.debug(
                "Parameters won't be saved! If you want to save them, set out_inference=True."
            )

        # If plot_loglik and flag_conv are both True, plot the best log-likelihood values
        if np.logical_and(self.plot_loglik, self.flag_conv == "log"):
            plot_L(best_loglik_values, int_ticks=True)

    def _log_realization_info(
        self,
        r,
        loglik,
        final_it,
        time_start,
        convergence,
    ) -> None:  # type: ignore
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
            "num. realizations = %s - Log-likelihood = %s - iterations = %s - time = %s seconds - "
            "convergence = %s",
            r,
            loglik,
            final_it,
            np.round(time.time() - time_start, 2),
            convergence,
        )

    @abstractmethod
    def compute_likelihood(self) -> float:
        """
        Compute the log-likelihood of the data.

        This is an abstract method that must be implemented in each derived class.
        """

    def load_data(self, files: str, adj_name: str, **kwargs: Any) -> GraphData:
        """
        Load the data from the input files.

        Parameters
        ----------
        files
            The directory containing the input files.
        adj_name
            The name of the adjacency file.
        **kwargs
            Additional keyword arguments to pass to the data loading function.

        Returns
        -------
        GraphData
            The loaded graph data.
        """
        file_path = os.path.join(files, adj_name)
        gdata: GraphData = build_adjacency_from_file(file_path, **kwargs)
        return gdata

    def get_params_to_load_data(self, args: Namespace) -> Dict[str, Any]:
        """
        Get the parameters from the arguments.

        Parameters
        ----------
        args : Namespace
               The arguments.

        Returns
        -------
        params : Dict[str, Any]
                 The parameters.
        """
        fields = [
            "files",
            "adj_name",
            "ego",
            "alter",
            "undirected",
            "force_dense",
            "noselfloop",
            "binary",
        ]
        return {f: getattr(args, f) for f in fields}

    @singledispatchmethod
    def get_data_sum(self, data: GraphDataType) -> float:
        """
        Compute the sum of the data.

        Parameters
        ----------
        data : GraphDataType
            The data to sum.

        Raises
        ------
        TypeError
            If the data type is unsupported.
        """
        raise TypeError("Unsupported data type")

    @get_data_sum.register
    def _(self, data: np.ndarray) -> float:
        """
        Compute the sum of a numpy array.

        Parameters
        ----------
        data : np.ndarray
            The numpy array to sum.

        Returns
        -------
        float
            The sum of the numpy array.
        """
        return data.sum()

    @get_data_sum.register
    def _(self, data: COO) -> float:
        """
        Compute the sum of a COO sparse array.

        Parameters
        ----------
        data : COO
            The COO sparse array to sum.

        Returns
        -------
        float
            The sum of the COO sparse array.
        """
        return data.data.sum()

    @singledispatchmethod
    def get_data_toarray(self, data: GraphDataType) -> np.ndarray:
        """
        Convert the data to a numpy array.

        Parameters
        ----------
        data : GraphDataType
            The data to convert.

        Returns
        -------
        np.ndarray
            The converted numpy array.

        Raises
        ------
        TypeError
            If the data type is unsupported.
        """
        raise TypeError("Unsupported data type")

    @get_data_toarray.register
    def _(self, data: np.ndarray) -> np.ndarray:
        """
        Return the numpy array as is.

        Parameters
        ----------
        data : np.ndarray
            The numpy array.

        Returns
        -------
        np.ndarray
            The same numpy array.
        """
        return data

    @get_data_toarray.register
    def _(self, data: COO) -> np.ndarray:
        """
        Convert a COO sparse array to a numpy array.

        Parameters
        ----------
        data : COO
            The COO sparse array to convert.

        Returns
        -------
        np.ndarray
            The converted numpy array.
        """
        return data.toarray()

    @singledispatchmethod
    def get_data_nonzero(self, data: GraphDataType) -> tuple:
        """
        Get the indices of non-zero elements in the data.

        Parameters
        ----------
        data : GraphDataType
            The data to get non-zero indices from.

        Returns
        -------
        tuple
            The indices of non-zero elements.

        Raises
        ------
        TypeError
            If the data type is unsupported.
        """
        raise TypeError("Unsupported data type")

    @get_data_nonzero.register
    def _(self, data: np.ndarray) -> tuple:
        """
        Get the indices of non-zero elements in a numpy array.

        Parameters
        ----------
        data : np.ndarray
            The numpy array to get non-zero indices from.

        Returns
        -------
        tuple
            The indices of non-zero elements.
        """
        return data.nonzero()

    @get_data_nonzero.register
    def _(self, data: COO) -> tuple:
        """
        Get the indices of non-zero elements in a COO sparse array.

        Parameters
        ----------
        data : COO
            The COO sparse array to get non-zero indices from.

        Returns
        -------
        tuple
            The indices of non-zero elements.
        """
        return data.coords

    @singledispatchmethod
    def get_data_values(self, data) -> np.ndarray:
        """
        Get the values of non-zero elements in the data.

        Parameters
        ----------
        data : Union[np.ndarray, COO]
            The data to get non-zero values from.

        Returns
        -------
        np.ndarray
            The values of non-zero elements.

        Raises
        ------
        TypeError
            If the data type is unsupported.
        """
        raise TypeError("Unsupported data type")

    @get_data_values.register
    def _(self, data: np.ndarray) -> np.ndarray:
        """
        Get the values of non-zero elements in a numpy array.

        Parameters
        ----------
        data : np.ndarray
            The numpy array to get non-zero values from.

        Returns
        -------
        np.ndarray
            The values of non-zero elements.
        """
        return data[data.nonzero()]

    @get_data_values.register
    def _(self, data: COO) -> np.ndarray:
        """
        Get the values of non-zero elements in a COO sparse array.

        Parameters
        ----------
        data : COO
            The COO sparse array to get non-zero values from.

        Returns
        -------
        np.ndarray
            The values of non-zero elements.
        """
        return data.data


class ModelUpdateMixin(ABC):
    """
    Mixin class for the update methods of the models classes. It is not a requirement to inherit
    from this class.
    """

    def __init__(self):
        self.max_iter = None
        self.err_max = None
        self.flag_conv = None
        self._check_for_convergence = None
        self._check_for_convergence_delta = None
        self.time_start = None
        self.u = None
        self.v = None
        self.w = None
        self.u_old = None
        self.v_old = None
        self.w_old = None
        self.delta_u = None
        self.delta_v = None
        self.delta_w = None
        self.delta_eta = None
        self.compute_likelihood = None

    def _finalize_update(
        self, matrix: np.ndarray, matrix_old: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        low_values_indices = matrix < self.err_max  # values are too low
        matrix[low_values_indices] = 0.0  # and set to 0.

        dist = np.amax(abs(matrix - matrix_old))
        matrix_old = np.copy(matrix)

        return dist, matrix, matrix_old

    def _update_U(self) -> float:
        # A generic function here that will do what each class needs
        self._specific_update_U()

        # Finalize the update of the membership matrix U
        dist, self.u, self.u_old = self._finalize_update(self.u, self.u_old)

        return dist

    @abstractmethod
    def _specific_update_U(self):
        """
        Update the membership matrix U.
        """

    def _update_V(self) -> float:
        # a generic function here that will do what each class needs
        self._specific_update_V()  # subs_nz, subs_X_nz, mask, subs_nz_mask)

        dist, self.v, self.v_old = self._finalize_update(self.v, self.v_old)

        return dist

    @abstractmethod
    def _specific_update_V(self):
        """
        Update the membership matrix V.

        This is an abstract method that must be implemented in each derived class.
        """

    def _update_W(self) -> float:
        # a generic function here that will do what each class needs
        self._specific_update_W()

        dist, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

        return dist

    def _specific_update_W(self, *args, **kwargs):
        """
        Update the affinity tensor W.

        This is an abstract method that must be implemented in each derived class.
        """

    def _update_W_assortative(self) -> float:
        # a generic function here that will do what each class needs

        self._specific_update_W_assortative()

        dist, self.w, self.w_old = self._finalize_update(self.w, self.w_old)

        return dist

    @abstractmethod
    def _specific_update_W_assortative(self):
        """
        Update the membership matrix.

        This is an abstract method that must be implemented in each derived class.
        """

    @abstractmethod
    def _update_membership(self, *args, **kwargs):
        """
        Update the membership matrix.

        This is an abstract method that must be implemented in each derived class.
        """

    @abstractmethod
    def _update_cache(self, *args, **kwargs):
        """
        Update the cache.

        This is an abstract method that must be implemented in each derived class.
        """

    def _update_old_variables(self) -> None:
        """
        Update values of the parameters in the previous iteration.
        """
        self._copy_variables(source_suffix="", target_suffix="_old")

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """
        self._copy_variables(source_suffix="", target_suffix="_f")

    def _update_realization(
        self,
        r,
        it,
        loglik,
        coincide,
        convergence,
        loglik_values,
    ) -> Tuple[int, float, int, bool, list]:
        """
        Perform the EM update and check for convergence.

        Parameters
        ----------
        r : int
            Number of realizations.
        it : int
            Number of iterations.
        loglik : float
            Log-likelihood value.
        coincide : int
            Number of times the update of the log-likelihood respects the convergence tolerance.
        convergence : bool
            Flag for convergence.
        loglik_values : list
            List of log-likelihood values.

        Returns
        -------
        it : int
            Updated number of iterations.
        loglik : float
            Updated log-likelihood value.
        coincide : int
            Updated number of times the update of the log-likelihood respects the convergence tolerance.
        convergence : bool
            Updated flag for convergence.
        loglik_values : list
            Updated list of log-likelihood values.
        """
        logging.debug("Updating realization %s ...", r)

        # It enters a while loop that continues until either convergence is achieved or the
        # maximum number of iterations (self.max_iter) is reached.
        while not convergence and it < self.max_iter:
            # It performs the main EM update (self._update_em()
            # which updates the memberships and calculates the maximum difference
            # between new and old parameters.
            self._update_em()
            # Depending on the convergence flag (self.flag_conv), it checks for convergence using
            # either the  log-likelihood values (self._check_for_convergence(data, it,
            # loglik,  coincide, convergence, data_T=data_T, mask=mask)) or the maximum distances
            # between the old and the new parameters (self._check_for_convergence_delta(it,
            # coincide, delta_u, delta_v, delta_w, delta_eta, convergence)).

            if self.flag_conv == "log":
                it, loglik, coincide, convergence = self._check_for_convergence(
                    it, loglik, coincide, convergence
                )
                loglik_values.append(loglik)
                if not it % 100:
                    logging.debug(
                        "num. realization = %s - iterations = %s - time = %.2f seconds",
                        r,
                        it,
                        time.time() - self.time_start,
                    )
            elif self.flag_conv == "deltas":
                it, coincide, convergence = self._check_for_convergence_delta(
                    it,
                    coincide,
                    self.delta_u,
                    self.delta_v,
                    self.delta_w,
                    self.delta_eta,
                    convergence,
                )

                if not it % 100:
                    logging.debug(
                        "Nreal = %s - iterations = %s - time = %.2f seconds",
                        r,
                        it,
                        time.time() - self.time_start,
                    )
            else:
                log_and_raise_error(
                    ValueError, "flag_conv can be either log or deltas!"
                )
        # After the while loop, it checks if the current  log-likelihood is the maximum
        # so far. If it is, it updates the optimal parameters (
        # self._update_optimal_parameters()) and sets maxL to the current log-likelihood.
        if self.flag_conv == "deltas":
            loglik = self.compute_likelihood()  # data, data_T, mask

        return it, loglik, coincide, convergence, loglik_values

    @abstractmethod
    def _update_em(self, *args, **kwargs):
        """
        Update parameters via EM procedure.

        This is an abstract method that must be implemented in each derived class.
        """

    def _copy_variables(self, source_suffix: str, target_suffix: str) -> None:
        # of derived classes
        """
        Copy variables from source to target.

        Parameters
        ----------
        source_suffix : str
                        The suffix of the source variable names.
        target_suffix : str
                        The suffix of the target variable names.
        """
        for var in ["u", "v", "w"]:
            source_var = getattr(self, f"{var}{source_suffix}")
            setattr(self, f"{var}{target_suffix}", np.copy(source_var))
