import dataclasses
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, Union

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

    def _check_fit_params(self,
                           initialization: int,
                           eta0: Union[float, None],
                           undirected: bool,
                           assortative: bool,
                           data: Union[skt.dtensor, skt.sptensor],
                           K: int,
                           message: str = None,
                           **extra_params: Unpack[FitParams]
                           ) -> None:
        # TODO: add a way of checking the existence of the extra params beloning to a subclass
        if initialization not in {0, 1, 2, 3}:
            log_and_raise_error(ValueError, message)

        self.initialization = initialization

        if (eta0 is not None) and (eta0 <= 0.):
            message = 'If not None, the eta0 parameter has to be greater than 0.!'
            log_and_raise_error(ValueError, message)

        self.eta0 = eta0
        self.undirected = undirected
        self.assortative = assortative

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
            'end_file',
            'use_approximation'
        ]

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

        # Initialize parameters
        self._initialize_parameters()

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

            logging.debug('eta is initialized randomly.')
            self._randomize_eta(use_unit_uniform=self.use_unit_uniform)

        if self.initialization == 0:

            logging.debug('%s', 'u, v and w are initialized randomly.')
            self._randomize_w()
            self._randomize_u_v(normalize_rows=self.normalize_rows)

        elif self.initialization == 1:

            logging.debug('%s is initialized using the input file: %s', 'w', self.files)
            logging.debug('%s', 'u and v are initialized randomly.')
            self._initialize_w()
            self._randomize_u_v(normalize_rows=self.normalize_rows)

        elif self.initialization == 2:

            logging.debug(
                'u and v are initialized using the input file: %s', self.files
            )
            logging.debug('%s', 'w is initialized randomly.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._randomize_w()

        elif self.initialization == 3:

            logging.debug(
                'u, v and w are initialized using the input file: %s', self.files
            )
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._initialize_w()


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

        if normalize_rows:
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            self.v = self.rng.random_sample((self.N, self.K))
            if normalize_rows:
                row_sums = self.v.sum(axis=1)
                self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
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
    
    def _update_old_variables(self) -> None:
        """
        Update values of the parameters in the previous iteration.
        """

        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        self.w_old = np.copy(self.w)
        self.eta_old = float(self.eta)

    def _update_optimal_parameters(self) -> None:
        """
        Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = float(self.eta)

    def _lambda_nz(self, subs_nz: tuple) -> np.ndarray:
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

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :],
                                    self.w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], 
                                    self.w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ,
                               self.v[subs_nz[2], :])

        return nz_recon_I

    def _check_for_convergence(self,
                               data: Union[skt.dtensor, skt.sptensor],
                               it: int,
                               loglik: float,
                               coincide: int,
                               convergence: bool,
                               use_pseudo_likelihood: bool = False,
                               data_T: Optional[skt.sptensor] = None,
                               mask: Optional[np.ndarray] = None) -> Tuple[
        int, float, int, bool]:
        """
        Check for convergence by using the log-likelihood values.

        Parameters
        ----------
        data : sptensor/dtensor
               Graph adjacency tensor.
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        use_pseudo_likelihood : bool
                                Flag to determine which log-likelihood function to use.
        data_T : sptensor/dtensor, optional
                 Graph adjacency tensor (transpose).
        mask : ndarray, optional
               Mask for selecting the held out set in the adjacency tensor in case of
               cross-validation.

        Returns
        -------
        it : int
             Number of iteration.
        loglik : float
                 Log-likelihood value.
        coincide : int
                   Number of time the update of the log-likelihood respects the convergence_tol.
        convergence : bool
                      Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            if use_pseudo_likelihood:
                loglik = self._PSLikelihood(data, data_T=data_T, mask=mask)
            else:
                loglik = self._Likelihood(data)
            if abs(loglik - old_L) < self.convergence_tol:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
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

        if (du < self.convergence_tol and dv < self.convergence_tol and dw < self.convergence_tol
                and de < self.convergence_tol):
            coincide += 1
        else:
            coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence
    def _output_results(self, max_val: float, nodes: List[Any]) -> None:
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
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        outfile = (Path(self.out_folder) / str('theta' + self.end_file)).with_suffix('.npz')
        np.savez_compressed(outfile,
                            u=self.u_f,
                            v=self.v_f,
                            w=self.w_f,
                            eta=self.eta_f,
                            max_it=self.final_it,
                            **{self.get_max_label(): max_val},
                            nodes=nodes)

        logging.info('Inferred parameters saved in: %s', outfile.resolve())
        logging.info('To load: theta=np.load(filename), then e.g. theta["u"]')


    def _initialize_parameters(self):
        pass
        #raise NotImplementedError("This method should be overridden in the subclass.")

