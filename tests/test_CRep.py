from pathlib import Path
import unittest

import numpy as np

from pgm.model.crep import CRep, sp_uttkrp, sp_uttkrp_assortative


class TestCRepHelpers(unittest.TestCase):

    def setUp(self):
        # Initialize a CRep object for testing
        self.N = 100
        self.L = 1
        self.K = 2
        self.crep = CRep()
        self.crep.N = self.N
        self.crep.L = self.L
        self.crep.K = self.K
        self.crep.rng = np.random.RandomState(0)
        self.crep.files = Path('pgm').resolve() /'data'/'input'/'theta_gt111.npz'
        self.crep.theta = np.load(self.crep.files,
                                  allow_pickle=True)  # TODO: use package data
        self.crep.eta0 = 0
        self.crep.undirected = False
        self.crep.assortative = True
        self.crep.constrained = True

        # Parameters for the non assortative case
        self.vals_ = np.array([1.0, 2.0, 3.0])
        self.subs_ = (np.array([0, 0, 1]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.u_ = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.v_ = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        self.w_ = np.array([[[0.1, 0.2], [0.3, 0.4]],
                            [[0.5, 0.6], [0.7, 0.8]],
                            [[0.9, 1.0], [1.1, 1.2]]])
        # Parameters for the assortative case
        self.vals_a = np.array(
            [1.0, 2.0,
             3.0])  # of size n, number of nodes; these are the non zero values of the tensor
        self.subs_a = (np.array([0, 0, 0]), np.array([0, 1, 2]),
                       np.array([0, 1, 2]))  # list of arrays of indices, each one of size n, lxixj
        self.u_a = np.array([[0.1, 0.2], [0.4, 0], [0.5, 0.6]])  # of size nxk,
        self.v_a = np.array([[0.7, 0], [0.9, 0], [1.1, 0]])  # of size n
        self.w_a = np.array([[0.1, 0.2, 0]])  # lxk #in the other case, this is lxkxk

    def test_randomize_eta(self):
        self.crep._randomize_eta()
        self.assertTrue(0 <= self.crep.eta <= 1)

    def test_randomize_w(self):

        self.crep._randomize_w()
        if self.crep.assortative:
            self.assertEqual(self.crep.w.shape, (self.crep.L, self.crep.K))
        else:
            self.assertEqual(self.crep.w.shape, (self.crep.L, self.crep.K, self.crep.K))

    def test_randomize_u_v(self):
        self.crep._randomize_u_v()
        row_sums_u = self.crep.u.sum(axis=1)
        self.assertTrue(np.all((0 <= self.crep.u) & (self.crep.u <= 1)))
        self.assertTrue(np.all(row_sums_u > 0))
        if not self.crep.undirected:
            row_sums_v = self.crep.v.sum(axis=1)
            self.assertTrue(np.all((0 <= self.crep.v) & (self.crep.v <= 1)))
            self.assertTrue(np.all(row_sums_v > 0))

    def test_initialize_random_eta(self):

        self.crep.initialization = 0
        self.crep._initialize(nodes=[0, 1, 2])
        self.assertTrue(0 <= self.crep.eta <= 1)

    def test_initialize_random_uvw(self):

        self.crep.initialization = 0
        self.crep._initialize(nodes=[0, 1, 2])
        self.assertTrue(np.all((0 <= self.crep.u) & (self.crep.u <= 1)))
        self.assertTrue(np.all((0 <= self.crep.v) & (self.crep.v <= 1)))
        self.assertTrue(np.all((0 <= self.crep.w) & (self.crep.w <= 1)))

    def test_initialize_w_from_file(self):
        self.crep.initialization = 1

        dfW = self.crep.theta['w']
        self.crep.L = dfW.shape[0]
        self.crep.K = dfW.shape[1]
        self.crep._initialize(nodes=[0, 1, 2])
        self.assertTrue(np.all(0 <= self.crep.w))

    def test_initialize_uv_from_file(self):
        self.crep.initialization = 2
        self.crep._initialize(nodes=range(600))  # Set by hand
        self.assertTrue(np.all(0 <= self.crep.u))
        self.assertTrue(np.all(0 <= self.crep.v))

    def test_initialize_uvw_from_file(self):
        self.crep.initialization = 3
        self.crep.L, self.crep.K = self.w_a.shape
        self.crep._initialize(nodes=range(600))
        self.assertTrue(np.all(0 <= self.crep.u))
        self.assertTrue(np.all(0 <= self.crep.v))
        self.assertTrue(np.all(0 <= self.crep.w))

    def test_sp_uttkrp_mode_1(self):
        mode = 1
        result = sp_uttkrp(self.vals_, self.subs_, mode, self.u_, self.v_, self.w_)
        expected_result = np.array([[0.23, 0.53], [0.58, 1.34], [3.81, 5.19]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_mode_2(self):
        mode = 2
        result = sp_uttkrp(self.vals_, self.subs_, mode, self.u_, self.v_, self.w_)
        expected_result = np.array([[0.07, 0.1], [0.3, 0.44], [2.01, 2.34]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_assortative_mode_1(self):
        mode = 1
        result = sp_uttkrp_assortative(self.vals_a, self.subs_a, mode, self.u_a, self.v_a,
                                       self.w_a)  # this is nxk
        expected_result = np.array([[0.07, 0.], [0.18, 0], [0.33, 0]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_assortative_mode_2(self):
        mode = 2
        result = sp_uttkrp_assortative(self.vals_a, self.subs_a, mode, self.u_a, self.v_a,
                                       self.w_a)  # this is nxk
        expected_result = np.array([[0.01, 0.04], [0.08, 0], [0.15, 0.36]])
        self.assertTrue(np.allclose(result, expected_result))
