"""
    Class definition of JointCRep, the algorithm to perform inference in networks with reciprocity.
    The latent variables are related to community memberships and a pair interaction value.
"""

from __future__ import print_function

import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sktensor as skt
from termcolor import colored


class JointCRep:
    def __init__(
            self,
            N=100,
            L=1,
            K=2,
            undirected=False,
            assortative=False,
            rseed=0,
            inf=1e10,
            err_max=1e-8,
            err=0.01,
            N_real=1,
            tolerance=0.0001,
            decision=10,
            max_iter=500,
            initialization=0,
            fix_eta=False,
            fix_communities=False,
            fix_w=False,
            use_approximation=False,
            eta0=None,
            files='../data/input/synthetic/theta.npz',
            verbose=False,
            out_inference=False,
            out_folder='../data/output/',
            end_file='.dat',
            plot_loglik=False):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities
        self.undirected = undirected  # flag to call the undirected network
        self.assortative = assortative  # if True, the network is assortative
        self.rseed = rseed  # random rseed for the initialization
        self.inf = inf  # initial value of the log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = N_real  # number of iterations with different random initialization
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.fix_eta = fix_eta  # if True, keep the eta parameter fixed
        self.fix_communities = fix_communities  # if True, keep the communities u and v fixed
        self.fix_w = fix_w  # if True, keep the affinity tensor fixed
        self.use_approximation = use_approximation  # if True, use the approximated version of the updates
        self.files = files  # path of the input files for u, v, w (when initialization>0)
        self.verbose = verbose  # flag to print details
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.end_file = end_file  # output file suffix
        self.plot_loglik = plot_loglik  # flag to plot the log-likelihood
        if initialization not in {
                0, 1, 2, 3}:  # indicator for choosing how to initialize u, v and w
            raise ValueError(
                'The initialization parameter can be either 0, 1, 2 or 3. It is used as an indicator to '
                'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
                'will be generated randomly; 1 means only the affinity matrix w will be uploaded from '
                'file; 2 implies the membership matrices u and v will be uploaded from file and 3 all u, '
                'v and w will be initialized through an input file.')
        self.initialization = initialization
        if (eta0 is not None) and (eta0 <= 0.):  # initial value for the pair interaction coefficient
            raise ValueError('If not None, the eta0 parameter has to be greater than 0.!')
        self.eta0 = eta0
        if self.fix_eta:
            if self.eta0 is None:
                raise ValueError('If fix_eta=True, provide a value for eta0.')
        if self.fix_w:
            if self.initialization not in {1, 3}:
                raise ValueError('If fix_w=True, the initialization has to be either 1 or 3.')
        if self.fix_communities:
            if self.initialization not in {2, 3}:
                raise ValueError(
                    'If fix_communities=True, the initialization has to be either 2 or 3.')

        if self.initialization > 0:
            self.theta = np.load(self.files, allow_pickle=True)
            if self.initialization == 1:
                dfW = self.theta['w']
                self.L = dfW.shape[0]
                self.K = dfW.shape[1]
            elif self.initialization == 2:
                dfU = self.theta['u']
                self.N, self.K = dfU.shape
            else:
                dfW = self.theta['w']
                dfU = self.theta['u']
                self.L = dfW.shape[0]
                self.K = dfW.shape[1]
                self.N = dfU.shape[0]
                assert self.K == dfU.shape[1]

        if self.undirected:
            if not (self.fix_eta and self.eta0 == 1):
                raise ValueError(
                    'If undirected=True, the parameter eta has to be fixed equal to 1 (s.t. log(eta)=0).')

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta = 0.  # pair interaction term

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_old = 0.  # pair interaction coefficient

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_f = np.zeros((self.N, self.K), dtype=float)  # in-going membership
        self.eta_f = 0.  # pair interaction coefficient

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

        if self.fix_eta:
            self.eta = self.eta_old = self.eta_f = self.eta0

    def fit(self, data, data_T, data_T_vals, nodes, flag_conv):
        """
            Model directed networks by using a probabilistic generative model based on a Bivariate Bernoulli
            distribution that assumes community parameters and a pair interaction coefficient as latent variables.
            The inference is performed via EM algorithm.

            Parameters
            ----------
            data : ndarray/sptensor
                   Graph adjacency tensor.
            data_T: None/sptensor
                    Graph adjacency tensor (transpose) - if sptensor.
            data_T_vals : None/ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j) - if ndarray.
            nodes : list
                    List of nodes IDs.
            flag_conv : str
                        If 'log' the convergence is based on the log-likelihood values; if 'deltas' the convergence is
                        based on the differences in the parameters values. The latter is suggested when the dataset
                        is big (N > 1000 ca.).

            Returns
            -------
            u_f : ndarray
                  Out-going membership matrix.
            v_f : ndarray
                  In-coming membership matrix.
            w_f : ndarray
                  Affinity tensor.
            eta_f : float
                    Pair interaction coefficient.
            maxL : float
                   Maximum log-likelihood.
        """

        if data_T is None:
            data_T = np.einsum('aij->aji', data)
            data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
            # pre-processing of the data to handle the sparsity
            data = preprocess(data)

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        self.AAtSum = (data.vals * data_T_vals).sum()

        rng = np.random.RandomState(self.rseed)

        maxL = -self.inf  # initialization of the maximum log-likelihood

        for r in range(self.N_real):

            self._initialize(rng=rng, nodes=nodes)

            self._update_old_variables()
            self._update_cache(data, subs_nz)
            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            if self.verbose:
                print(f'Updating realization {r} ...')
            loglik_values = []
            time_start = time.time()
            # --- single step iteration update ---
            while np.logical_and(not convergence, it < self.max_iter):
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w, delta_eta = self._update_em(data, subs_nz)
                if flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(
                        data, it, loglik, coincide, convergence)
                    loglik_values.append(loglik)
                    if self.verbose:
                        if not it % 100:
                            print(f'Nreal = {r} - Log-likelihood = {loglik} - iterations = {it} - '
                                  f'time = {np.round(time.time() - time_start, 2)} seconds')
                elif flag_conv == 'deltas':
                    it, coincide, convergence = self._check_for_convergence_delta(
                        it, coincide, delta_u, delta_v, delta_w, delta_eta, convergence)
                    if self.verbose:
                        if not it % 100:
                            print(f'Nreal = {r} - iterations = {it} - '
                                  f'time = {np.round(time.time() - time_start, 2)} seconds')
                else:
                    raise ValueError('flag_conv can be either log or deltas!')

            if flag_conv == 'log':
                if maxL < loglik:
                    self._update_optimal_parameters()
                    best_loglik = list(loglik_values)
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    best_r = r
            elif flag_conv == 'deltas':
                loglik = self._Likelihood(data)
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    final_it = it
                    conv = convergence
                    best_r = r
            if self.verbose:
                print(f'Nreal = {r} - Log-likelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds\n')
            # print(f'Best real = {best_r} - maxL = {maxL} - best iterations = {final_it}')

            # end cycle over realizations

        print(f'Best real = {best_r} - maxL = {maxL} - best iterations = {final_it}')

        if np.logical_and(final_it == self.max_iter, not conv):
            # convergence not reaches
            try:
                print(
                    colored(
                        'Solution failed to converge in {0} EM steps!'.format(
                            self.max_iter),
                        'blue'))
            except BaseException:
                print('Solution failed to converge in {0} EM steps!'.format(self.max_iter))

        if np.logical_and(self.plot_loglik, flag_conv == 'log'):
            plot_L(best_loglik, int_ticks=True)

        if self.out_inference:
            self._output_results(maxL, nodes, final_it)

        return self.u_f, self.v_f, self.w_f, self.eta_f, maxL

    def _initialize(self, rng, nodes):
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
            if self.verbose:
                print('eta is initialized randomly.')
            self._randomize_eta(rng=rng)

        if self.initialization == 0:
            if self.verbose:
                print('u, v and w are initialized randomly.')
            self._randomize_w(rng=rng)
            self._randomize_u_v(rng=rng)

        elif self.initialization == 1:
            if self.verbose:
                print(f'w is initialized using the input file: {self.files}.')
                print('u and v are initialized randomly.')
            self._initialize_w()
            self._randomize_u_v(rng=rng)

        elif self.initialization == 2:
            if self.verbose:
                print(f'u and v are initialized using the input file: {self.files}.')
                print('w is initialized randomly.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._randomize_w(rng=rng)

        elif self.initialization == 3:
            if self.verbose:
                print(f'u, v and w are initialized using the input file: {self.files}.')
            self._initialize_u(nodes)
            self._initialize_v(nodes)
            self._initialize_w()

    def _randomize_eta(self, rng):
        """
            Generate a random number in (1., 50.).

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        self.eta = rng.uniform(1.01, 49.99)

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor w.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)

    def _randomize_u_v(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        self.u = rng.random_sample(self.u.shape)
        # row_sums = self.u.sum(axis=1)
        # self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        if not self.undirected:
            self.v = rng.random_sample(self.v.shape)
            # row_sums = self.v.sum(axis=1)
            # self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _initialize_u(self, nodes):
        """
            Initialize out-going membership matrix u from file.

            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        self.u = self.theta['u']
        assert np.array_equal(nodes, self.theta['nodes'])

        max_entry = np.max(self.u)
        self.u += max_entry * self.err * np.random.random_sample(self.u.shape)

    def _initialize_v(self, nodes):
        """
            Initialize in-coming membership matrix v from file.

            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        if self.undirected:
            self.v = self.u
        else:
            self.v = self.theta['v']
            assert np.array_equal(nodes, self.theta['nodes'])

            max_entry = np.max(self.v)
            self.v += max_entry * self.err * np.random.random_sample(self.v.shape)

    def _initialize_w(self):
        """
            Initialize affinity tensor w from file.
        """

        if self.assortative:
            self.w = self.theta['w']
            assert self.w.shape == (self.L, self.K)
        else:
            self.w = self.theta['w']

        max_entry = np.max(self.w)
        self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
        self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
        self.w_old[self.w > 0] = np.copy(self.w[self.w > 0])
        self.eta_old = np.copy(self.eta)

    def _update_cache(self, data, subs_nz):
        """
            Update the cache used in the em_update.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
        """

        self.lambda_aij = self._lambda_full()  # full matrix lambda

        self.lambda_nz = self._lambda_nz(subs_nz)  # matrix lambda for non-zero entries
        lambda_zeros = self.lambda_nz == 0
        self.lambda_nz[lambda_zeros] = 1  # still good because with np.log(1)=0
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.lambda_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.lambda_nz
        self.data_M_nz[lambda_zeros] = 0  # to use in the udpates

        self.den_updates = 1 + self.eta * self.lambda_aij  # to use in the updates
        if not self.use_approximation:
            self.lambdalambdaT = np.einsum(
                'aij,aji->aij',
                self.lambda_aij,
                self.lambda_aij)  # to use in Z and eta
            self.Z = self._calculate_Z()

    def _lambda_full(self):
        """
            Compute the mean lambda for all entries.

            Returns
            -------
            M : ndarray
                Mean lambda for all entries.
        """

        if self.w.ndim == 2:
            M = np.einsum('ik,jk->ijk', self.u, self.v)
            M = np.einsum('ijk,ak->aij', M, self.w)
        else:
            M = np.einsum('ik,jq->ijkq', self.u, self.v)
            M = np.einsum('ijkq,akq->aij', M, self.w)

        return M

    def _lambda_nz(self, subs_nz):
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
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :], self.w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, self.v[subs_nz[2], :])

        return nz_recon_I

    def _calculate_Z(self):
        """
            Compute the normalization constant of the Bivariate Bernoulli distribution.

            Returns
            -------
            Z : ndarray
                Normalization constant Z of the Bivariate Bernoulli distribution.
        """

        Z = self.lambda_aij + transpose_tensor(self.lambda_aij) + self.eta * self.lambdalambdaT + 1
        for l in range(len(Z)):
            assert check_symmetric(Z[l])

        return Z

    def _update_em(self, data, subs_nz):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
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
            d_eta : float
                    Maximum distance between the old and the new pair interaction coefficient eta.
        """

        if not self.fix_communities:
            if self.use_approximation:
                d_u = self._update_U_approx(subs_nz)
            else:
                d_u = self._update_U(subs_nz)
            self._update_cache(data, subs_nz)
        else:
            d_u = 0.

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
            self._update_cache(data, subs_nz)
        else:
            if not self.fix_communities:
                if self.use_approximation:
                    d_v = self._update_V_approx(subs_nz)
                else:
                    d_v = self._update_V(subs_nz)
                self._update_cache(data, subs_nz)
            else:
                d_v = 0.

        if not self.fix_w:
            if not self.assortative:
                if self.use_approximation:
                    d_w = self._update_W_approx(subs_nz)
                else:
                    d_w = self._update_W(subs_nz)
            else:
                if self.use_approximation:
                    d_w = self._update_W_assortative_approx(subs_nz)
                else:
                    d_w = self._update_W_assortative(subs_nz)
            self._update_cache(data, subs_nz)
        else:
            d_w = 0.

        if not self.fix_eta:
            self.lambdalambdaT = np.einsum(
                'aij,aji->aij',
                self.lambda_aij,
                self.lambda_aij)  # to use in Z and eta
            if self.use_approximation:
                d_eta = self._update_eta_approx()
            else:
                d_eta = self._update_eta()
            self._update_cache(data, subs_nz)
        else:
            d_eta = 0.

        return d_u, d_v, d_w, d_eta

    def _update_U_approx(self, subs_nz):
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

        self.u *= self._update_membership(subs_nz, 1)

        if not self.assortative:
            VW = np.einsum('jq,akq->ajk', self.v, self.w)
        else:
            VW = np.einsum('jk,ak->ajk', self.v, self.w)
        den = np.einsum('aji,ajk->ik', self.den_updates, VW)

        non_zeros = den > 0.
        self.u[den == 0] = 0.
        self.u[non_zeros] /= den[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

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

        self.u *= self._update_membership(subs_nz, 1)

        if not self.assortative:
            VW = np.einsum('jq,akq->ajk', self.v, self.w)
        else:
            VW = np.einsum('jk,ak->ajk', self.v, self.w)
        VWL = np.einsum('aji,ajk->aijk', self.den_updates, VW)
        den = np.einsum('aijk,aij->ik', VWL, 1. / self.Z)

        non_zeros = den > 0.
        self.u[den == 0] = 0.
        self.u[non_zeros] /= den[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V_approx(self, subs_nz):
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

        self.v *= self._update_membership(subs_nz, 2)

        if not self.assortative:
            UW = np.einsum('jq,aqk->ajk', self.u, self.w)
        else:
            UW = np.einsum('jk,ak->ajk', self.u, self.w)
        den = np.einsum('aij,ajk->ik', self.den_updates, UW)

        non_zeros = den > 0.
        self.v[den == 0] = 0.
        self.v[non_zeros] /= den[non_zeros]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

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

        self.v *= self._update_membership(subs_nz, 2)

        if not self.assortative:
            UW = np.einsum('jq,aqk->ajk', self.u, self.w)
        else:
            UW = np.einsum('jk,ak->ajk', self.u, self.w)
        UWL = np.einsum('aij,ajk->aijk', self.den_updates, UW)
        den = np.einsum('aijk,aij->ik', UWL, 1. / self.Z)

        non_zeros = den > 0.
        self.v[den == 0] = 0.
        self.v[non_zeros] /= den[non_zeros]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_W_approx(self, subs_nz):
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

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0],
                                                   weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        den = np.einsum('jq,aijk->akq', self.v, UL)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative_approx(self, subs_nz):
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

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        den = np.einsum('jk,aijk->ak', self.v, UL)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

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

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
        for k in range(self.K):
            for q in range(self.K):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0],
                                                   weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        num = np.einsum('jq,aijk->aijkq', self.v, UL)
        den = np.einsum('aijkq,aij->akq', num, 1. / self.Z)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

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

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w = self.w_old * uttkrp_DKQ

        UL = np.einsum('ik,aji->aijk', self.u, self.den_updates)
        num = np.einsum('jk,aijk->aijk', self.v, UL)
        den = np.einsum('aijk,aij->ak', num, 1. / self.Z)

        non_zeros = den > 0.
        self.w[den == 0] = 0.
        self.w[non_zeros] /= den[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_eta_approx(self):
        """
            Update pair interaction coefficient eta.

            Returns
            -------
            dist_eta : float
                       Maximum distance between the old and the new pair interaction coefficient eta.
        """

        den = self.lambdalambdaT.sum()
        if not den > 0.:
            raise ValueError('eta update_approx has zero denominator!')
        else:
            self.eta = self.AAtSum / den

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)

        return dist_eta

    def eta_fix_point(self):
        st = (self.lambdalambdaT / self.Z).sum()
        if st > 0:
            return self.AAtSum / st
        else:
            print(self.lambdalambdaT, self.Z)
            raise ValueError('eta fix point has zero denominator!')

    def _update_eta(self):
        """
            Update pair interaction coefficient eta.

            Returns
            -------
            dist_eta : float
                       Maximum distance between the old and the new pair interaction coefficient eta.
        """

        self.eta = self.eta_fix_point()

        if self.eta < self.err_max:  # value is too low
            self.eta = 0.  # and set to 0.

        dist_eta = abs(self.eta - self.eta_old)
        self.eta_old = np.copy(self.eta)

        return dist_eta

    def _update_membership(self, subs_nz, m):
        """
            Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
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
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, self.u, self.v, self.w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m, self.u, self.v, self.w)

        return uttkrp_DK

    def _check_for_convergence(self, data, it, loglik, coincide, convergence):
        """
            Check for convergence by using the log-likelihood values.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            it : int
                 Number of iteration.
            loglik : float
                     Pseudo log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            loglik = self._Likelihood(data)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def _check_for_convergence_delta(self, it, coincide, du, dv, dw, de, convergence):
        """
            Check for convergence by using the maximum distances between the old and the new parameters values.

            Parameters
            ----------
            it : int
                 Number of iteration.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
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
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if du < self.tolerance and dv < self.tolerance and dw < self.tolerance and de < self.tolerance:
            coincide += 1
        else:
            coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _Likelihood(self, data):
        """
            Compute the log-likelihood of the data.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.

            Returns
            -------
            l : float
                Log-likelihood value.
        """

        self.lambdalambdaT = np.einsum(
            'aij,aji->aij',
            self.lambda_aij,
            self.lambda_aij)  # to use in Z and eta
        self.Z = self._calculate_Z()

        ft = (data.vals * np.log(self.lambda_nz)).sum()

        st = 0.5 * np.log(self.eta) * self.AAtSum

        tt = 0.5 * np.log(self.Z).sum()

        l = ft + st - tt

        if np.isnan(l):
            raise ValueError('log-likelihood is NaN!')
        else:
            return l

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.eta_f = np.copy(self.eta)

    def _output_results(self, maxL, nodes, final_it):
        """
            Output results.

            Parameters
            ----------
            maxL : float
                   Maximum log-likelihood.
            nodes : list
                    List of nodes IDs.
            final_it : int
                       Total number of iterations.
        """
        # Check if the output folder exists, otherwise create it
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        # Check if the output folder is a Path
        if not isinstance(self.out_folder, Path):
            self.out_folder = Path(self.out_folder)

        # outfile = self.out_folder + 'theta' + self.end_file
        outfile = (self.out_folder / str('theta' + self.end_file)).with_suffix('.npz')
        np.savez_compressed(outfile,
                            u=self.u_f,
                            v=self.v_f,
                            w=self.w_f,
                            eta=self.eta_f,
                            max_it=final_it,
                            maxL=maxL,
                            nodes=nodes)
        print(f'\nInferred parameters saved in: {outfile.resolve()}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) *
                    v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating v
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) *
                    u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
        elif m == 2:  # we are updating v
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def get_item_array_from_subs(A, ref_subs):
    """
        Get values of ref_subs entries of a dense tensor.
        Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(A):
    """
        Pre-process input data tensor.
        If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

        Parameters
        ----------
        A : ndarray
            Input data (tensor).

        Returns
        -------
        A : sptensor/dtensor
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not A.dtype == np.dtype(int).type:
        A = A.astype(int)
    if np.logical_and(isinstance(A, np.ndarray), is_sparse(A)):
        A = sptensor_from_dense_array(A)
    else:
        A = skt.dtensor(A)

    return A


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def plot_L(values, indices=None, k_i=5, figsize=(7, 7), int_ticks=False, xlab='Iterations'):
    """
        Plot the log-likelihood.
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel('Log-likelihood values')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
        Check if a matrix a is symmetric.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def transpose_tensor(M):
    """
        Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.

        Parameters
        ----------
        M : ndarray
            Tensor with the mean lambda for all entries.

        Returns
        -------
        Transpose version of M_aij, i.e. M_aji.
    """

    return np.einsum('aij->aji', M)
