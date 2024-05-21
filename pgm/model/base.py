import dataclasses
from functools import singledispatchmethod
import logging
from pathlib import Path
from typing import List, Tuple, TypedDict, Union

import numpy as np
import sktensor as skt
from typing_extensions import Unpack

from pgm.input.tools import log_and_raise_error


class FitParams(TypedDict):
    out_inference: bool
    out_folder: str
    end_file: str
    files: str
    fix_eta: bool
    fix_communities: bool
    fix_w: bool
    use_approximation: bool


@dataclasses.dataclass
class DataBase:
    inf: float = 1e10  # initial value of the log-likelihood
    err_max: float = 1e-12  # minimum value for the parameters
    err: float = 0.1  # noise for the initialization
    num_realizations: int = 3  # number of iterations with different random initialization
    convergence_tol: float = 0.0001  # convergence_tol parameter for convergence
    decision: int = 10  # convergence parameter
    max_iter: int = 500  # maximum number of EM steps before aborting
    plot_loglik: bool = False  # flag to plot the log-likelihood
    flag_conv: str = 'log'  # flag to choose the convergence criterion


class ModelClass(DataBase):
    def __init__(
            self,
            inf: float = 1e10,
            err_max: float = 1e-12,
            err: float = 0.1,
            num_realizations: int = 3,
            convergence_tol: float = 0.0001,
            decision: int = 10,
            max_iter: int = 500,
            plot_loglik: bool = False,
            flag_conv: str = 'log'):
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

        self.attributes_to_save_names = [
            'u_f',
            'v_f',
            'w_f',
            'eta_f',
            'final_it',
            'maxL',
            'maxPSL',
            'beta_f',
            'nodes']

    def _check_fit_params(self,
                          initialization: int,
                          undirected: bool,
                          assortative: bool,
                          data: Union[skt.dtensor, skt.sptensor],
                          K: int,
                          available_extra_params: List[str],
                          data_X: Union[skt.dtensor, skt.sptensor, np.ndarray, None],
                          eta0: Union[float, None],
                          beta0: Union[float, None],
                          gamma: Union[float, None],
                          message: str = "Invalid initialization parameter.",
                          **extra_params: Unpack[FitParams]
                          ) -> None:
        """
        Check the parameters of the fit method.
        """
        if initialization not in {0, 1}:
            log_and_raise_error(ValueError, message)

        self.initialization = initialization

        if (eta0 is not None) and (eta0 <= 0.):
            message = 'If not None, the eta0 parameter has to be greater than 0.!'
            log_and_raise_error(ValueError, message)

        if gamma is None:
            self.eta0 = eta0
        self.undirected = undirected
        self.assortative = assortative

        self.N = data.shape[1]
        self.L = data.shape[0]
        self.K = K
        if data_X is not None:
            self.gamma = gamma
            self.Z = data_X.shape[1]  # number of categories of the categorical attribute

        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = f'Ignoring extra parameter {extra_param}.'
                logging.warning(msg)

        if "fix_eta" in extra_params:
            self.fix_eta = extra_params["fix_eta"]

            if self.fix_eta:
                if self.eta0 is None:
                    log_and_raise_error(ValueError, 'If fix_eta=True, provide a value for eta0.')
        else:
            self.fix_eta = False

        if "fix_beta" in extra_params:
            self.fix_beta = extra_params["fix_beta"]

            if self.fix_beta:
                if beta0 is None:
                    log_and_raise_error(ValueError, 'If fix_beta=True, provide a value for beta0.')
                else:
                    self.beta0 = beta0

        if "fix_w" in extra_params:
            self.fix_w = extra_params["fix_w"]
            if self.fix_w:
                if self.initialization not in {1, 3}:
                    message = 'If fix_w=True, the initialization has to be either 1 or 3.'
                    log_and_raise_error(ValueError, message)
        else:
            self.fix_w = False

        if "fix_communities" in extra_params:
            self.fix_communities = extra_params["fix_communities"]
            if self.fix_communities:
                if self.initialization not in {2, 3}:
                    message = 'If fix_communities=True, the initialization has to be either 2 or 3.'
                    log_and_raise_error(ValueError, message)
        else:
            self.fix_communities = False

        if "files" in extra_params:
            self.files = extra_params["files"]

        if self.undirected and not (self.fix_eta and self.eta0 == 1):
            message = ('If undirected=True, the parameter eta has to be fixed equal to 1 '
                       '(s.t. log(eta)=0).')
            log_and_raise_error(ValueError, message)
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
            if not (self.fix_eta and self.eta0 == 1):
                message = ('If undirected=True, the parameter eta has to be fixed equal to 1.')
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

    def _initialize_eta(self) -> None:

        # If eta0 is not None, assign its value to eta
        if self.eta0 is not None:
            self.eta = self.eta0
        else:
            # If eta0 is None, log a message and call the method to randomize eta
            logging.debug('eta is initialized randomly.')
            self._randomize_eta(use_unit_uniform=self.use_unit_uniform)

    def _initialize_beta(self) -> None:
        """
        Placeholder function for initializing beta.
        Intentionally left empty for subclasses to override if necessary.
        """
        pass

    def _initialize_beta_from_file(self) -> None:
        # Assign the beta matrix from the input file to the beta attribute
        self.beta = self.theta['beta']

        # Add random noise to the beta matrix
        self.beta = self._add_random_noise(self.beta)

    def _random_initialization(self) -> None:
        # Log a message indicating that u, v and w are being initialized randomly
        logging.debug('%s', 'u, v and w are initialized randomly.')

        # Randomize w and u, v
        self._randomize_w()
        self._randomize_u_v(normalize_rows=self.normalize_rows)

    def _file_initialization(self) -> None:
        # Log a message indicating that u, v and w are being initialized using the input file
        logging.debug('u, v and w are initialized using the input file: %s', self.files)
        # Initialize u and v
        self._initialize_u()
        self._initialize_v()
        self._initialize_w()

    def _initialize_membership_matrix(self, matrix_name: str, matrix_value: np.ndarray) -> None:

        # Assign the input matrix value to the local variable 'matrix'
        matrix = matrix_value

        # Assert that the nodes in the current object and the nodes in the theta dictionary are the same
        # If they are not the same, raise an AssertionError with the message 'Nodes do not match.'
        assert np.array_equal(self.nodes, self.theta['nodes']), 'Nodes do not match.'

        # Find the maximum value in the 'matrix'
        max_entry = np.max(matrix)

        # Add random noise to the 'matrix'. The noise is a random number between 0 and 1,
        # multiplied by the maximum entry in the 'matrix' and the error rate 'self.err'
        matrix += max_entry * self.err * self.rng.random_sample(matrix.shape)

        # Set the attribute of the current object with the name 'matrix_name' to
        # the value of 'matrix'
        setattr(self, matrix_name, matrix)

    def _initialize_u(self) -> None:
        """
        Initialize out-going membership matrix u from file.
        """
        self._initialize_membership_matrix('u', self.theta['u'])

    def _initialize_v(self) -> None:
        """
        Initialize in-coming membership matrix v from file.
        """
        if self.undirected:
            self.v = self.u
        else:
            self._initialize_membership_matrix('v', self.theta['v'])

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
        matrix += max_entry * self.err * self.rng.random_sample(matrix.shape)
        return matrix

    def _initialize_w(self) -> None:
        """
        Initialize affinity tensor w from file.

        Parameters
        ----------
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.
        """
        self.w = self.theta['w']
        if self.assortative:
            assert self.w.shape == (self.L, self.K)

        self.w = self._add_random_noise(self.w)

    def _initialize_w_dyn(self):

        # Initialize the affinity tensor w from the input file
        w0 = self.theta['w']
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
        w0 = self.theta['w']
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
    def _randomize_beta(self, shape):
        """
        Generate a random number in (0, 1.).
        Parameters
        ----------
        shape : int
                The shape of the beta matrix.
        """
        self.beta = self.rng.random_sample(shape)

    @_randomize_beta.register
    def _(self, shape: tuple):
        """
        Assign a random number in (0, 1.) to each entry of the beta matrix, and normalize each row.
        Parameters
        ----------
        shape : tuple
                The shape of the beta matrix.
        """
        self.beta = self.rng.random_sample(shape)
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
        self.u = self.rng.random_sample((self.N, self.K))
        # Normalize each row of the membership matrix u
        if normalize_rows:
            # Compute the sum of each row of the membership matrix u
            row_sums = self.u.sum(axis=1)
            # Normalize each row of the membership matrix u if the sum of the row is greater than 0
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            # Set the membership matrix v to be a random sample of shape (self.N, self.K)
            self.v = self.rng.random_sample((self.N, self.K))
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
            self.w = self.rng.random_sample((self.L, self.K))
        else:
            self.w = self.rng.random_sample((self.L, self.K, self.K))

    def _randomize_eta(self, use_unit_uniform: bool = False) -> None:
        """
        Generate a random number in (0, 1.) or (1., 50.) based on the flag.

        Parameters
        ----------
        use_unit_uniform : bool
            If True, generate a random number in (1., 50.).
            If False, generate a random number in (0, 1.).
        """
        if use_unit_uniform:
            self.eta = float((self.rng.random_sample(1)[0]))
        else:
            self.eta = self.rng.uniform(1.01, 49.99)

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
        for var in ['u', 'v', 'w']:
            source_var = getattr(self, f"{var}{source_suffix}")
            setattr(self, f"{var}{target_suffix}", np.copy(source_var))

        if 'MTCOV' in type(self).__name__:
            source_var = getattr(self, f"beta{source_suffix}")
            setattr(self, f"beta{target_suffix}", np.copy(source_var))
        else:
            source_var = getattr(self, f"eta{source_suffix}")
            setattr(self, f"eta{target_suffix}", float(source_var))
            if 'DynCRep' in type(self).__name__:
                source_var = getattr(self, f"beta{source_suffix}")
                setattr(self, f"beta{target_suffix}", np.copy(source_var))

    def _update_old_variables(self) -> None:
        """
        Update values of the parameters in the previous iteration.
        """
        self._copy_variables(source_suffix='', target_suffix='_old')

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """
        self._copy_variables(source_suffix='', target_suffix='_f')
        if 'DynCRep' in type(self).__name__ and not self.fix_beta:
            self.beta_f = np.copy(self.beta_hat[-1])

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
                nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :],
                                        self.w[subs_nz[0], :, :])
            else:
                nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :],
                                        self.w[subs_nz[0], :])

        else:
            if not self.assortative:
                nz_recon_IQ = np.einsum('Ik,kq->Iq', self.u[subs_nz[1], :], self.w[0, :, :])
            else:
                nz_recon_IQ = np.einsum('Ik,k->Ik', self.u[subs_nz[1], :], self.w[0, :])

        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ,
                               self.v[subs_nz[2], :])

        return nz_recon_I

    def _check_for_convergence(self,
                               data,
                               it: int,
                               loglik: float,
                               coincide: int,
                               convergence: bool,
                               use_pseudo_likelihood: bool = False,
                               data_T_vals=None,
                               subs_nz=None,
                               T=None,
                               r=None,
                               data_T=None,
                               mask=None):
        """
        Check for convergence by using the log-likelihood values or the pseudo log-likelihood values.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value or Pseudo log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        use_pseudo_likelihood : bool
                                Flag to determine which log-likelihood function to use.
        data_T_vals : ndarray, optional
                      Values of the transpose of the adjacency tensor.
        subs_nz : tuple, optional
                  Indices of elements of data that are non-zero.
        T : int, optional
            Number of time steps.
        r : int, optional
            Number of realizations.
        data_T : sptensor/dtensor, optional
                 Graph adjacency tensor (transpose).
        mask : ndarray, optional
               Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

        Returns
        -------
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value or Pseudo log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        """
        # Check for convergence
        if it % 10 == 0:
            old_L = loglik
            if use_pseudo_likelihood:
                loglik = self._PSLikelihood(data, data_T=data_T, mask=mask)
            else:
                if data_T_vals is not None and subs_nz is not None and T is not None:
                    loglik = self._Likelihood(data, data_T, data_T_vals, subs_nz, T, mask=mask)
                else:
                    loglik = self._Likelihood(data)
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        # Define the convergence criterion
        convergence = coincide > self.decision or convergence
        # Update the number of iterations
        it += 1

        return it, loglik, coincide, convergence

    def _check_for_convergence_delta(self,
                                     it: int,
                                     coincide: int,
                                     du: float,
                                     dv: float,
                                     dw: float,
                                     de: float,
                                     convergence: bool) -> Tuple[int,
                                                                 int,
                                                                 bool]:
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

        if (du < self.convergence_tol
                and dv < self.convergence_tol
                and dw < self.convergence_tol
                and de < self.convergence_tol):
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
        # Check if the output folder exists, otherwise create it
        output_path = Path(self.out_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define the output file
        outfile = (Path(self.out_folder) / str('theta' + self.end_file)).with_suffix('.npz')

        # Create a dictionary to hold the attributes to be saved
        attributes_to_save = {}

        # Iterate over the instance's attributes
        for attr_name, attr_value in self.__dict__.items():
            # Check if the attribute is a numpy array and its name is in the list
            if attr_name in self.attributes_to_save_names:
                # Remove the '_f' suffix from the attribute name if it exists
                attr_name_clean = attr_name.removesuffix('_f')
                # Add the attribute to the dictionary with the cleaned name
                attributes_to_save[attr_name_clean] = attr_value

        # Save the attributes
        np.savez_compressed(outfile, **attributes_to_save)

        logging.info('Inferred parameters saved in: %s', outfile.resolve())
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')
