"""
Unit tests for the ModelClass class.
"""

import unittest
from importlib.resources import files
from pathlib import Path

import numpy as np

from probinet.models.base import ModelBase


class TestModelClass(unittest.TestCase):
    """
    Test cases for the ModelClass class.
    """

    def setUp(self):
        # Initialize a ModelClass object for testing
        self.N = 100
        self.L = 1
        self.K = 2
        self.model_class = ModelBase()
        self.model_class.N = self.N
        self.model_class.L = self.L
        self.model_class.K = self.K
        self.model_class.rng = np.random.RandomState(0)  # pylint: disable=no-member
        self.model_class.files = (
            Path("probinet").resolve() / "data" / "input" / "theta_gt111.npz"
        )
        self.model_class.theta_name = "theta_gt111.npz"
        with files("probinet.data.input").joinpath(
            self.model_class.theta_name
        ) as theta:
            self.model_class.theta = np.load(theta, allow_pickle=True)
        self.model_class.eta0 = 0
        self.model_class.undirected = False
        self.model_class.assortative = True
        self.model_class.constrained = True

        # Parameters for the initialization of the models
        self.model_class.use_unit_uniform = True
        self.model_class.normalize_rows = True

    def test_randomize_eta(self):
        self.model_class._randomize_eta(
            use_unit_uniform=True
        )  # pylint: disable=protected-access
        self.assertTrue(0 <= self.model_class.eta <= 1)

    def test_randomize_w(self):
        self.model_class._randomize_w()  # pylint: disable=protected-access
        if self.model_class.assortative:
            self.assertEqual(
                self.model_class.w.shape, (self.model_class.L, self.model_class.K)
            )
        else:
            self.assertEqual(
                self.model_class.w.shape,
                (self.model_class.L, self.model_class.K, self.model_class.K),
            )

    def test_randomize_u_v(self):
        self.model_class._randomize_u_v()  # pylint: disable=protected-access
        row_sums_u = self.model_class.u.sum(axis=1)
        self.assertTrue(np.all((0 <= self.model_class.u) & (self.model_class.u <= 1)))
        self.assertTrue(np.all(row_sums_u > 0))
        if not self.model_class.undirected:
            row_sums_v = self.model_class.v.sum(axis=1)
            self.assertTrue(
                np.all((0 <= self.model_class.v) & (self.model_class.v <= 1))
            )
            self.assertTrue(np.all(row_sums_v > 0))

    def test_initialize_random_eta(self):
        self.model_class.initialization = 0
        self.model_class._initialize()  # nodes=[0, 1, 2])  # pylint: disable=protected-access
        self.assertTrue(0 <= self.model_class.eta <= 1)

    def test_initialize_random_uvw(self):
        self.model_class.initialization = 0
        self.model_class._initialize()  # nodes=[0, 1, 2])  # pylint: disable=protected-access
        self.assertTrue(np.all((0 <= self.model_class.u) & (self.model_class.u <= 1)))
        self.assertTrue(np.all((0 <= self.model_class.v) & (self.model_class.v <= 1)))
        self.assertTrue(np.all((0 <= self.model_class.w) & (self.model_class.w <= 1)))

    def test_initialize_w_from_file(self):
        self.model_class.initialization = 1
        dfW = self.model_class.theta["w"]
        self.model_class.nodes = range(self.model_class.theta["nodes"].shape[0])
        self.model_class.L = dfW.shape[0]
        self.model_class.K = dfW.shape[1]
        self.model_class._initialize()  # pylint: disable=protected-access
        self.assertTrue(np.all(0 <= self.model_class.w))

    @unittest.skip("Deciding whether initialization 2 is useful or not.")
    def test_initialize_uv_from_file(self):
        self.model_class.initialization = 2
        self.model_class._initialize()  # pylint: disable=protected-access # Set by hand
        self.assertTrue(np.all(0 <= self.model_class.u))
        self.assertTrue(np.all(0 <= self.model_class.v))

    @unittest.skip("Deciding whether initialization 3 is useful or not.")
    def test_initialize_uvw_from_file(self):
        self.model_class.initialization = 3
        self.model_class.L, self.model_class.K = self.w_a.shape
        self.model_class._initialize()  # in case it is: nodes=range(600) # pylint: disable=protected-access
        self.assertTrue(np.all(0 <= self.model_class.u))
        self.assertTrue(np.all(0 <= self.model_class.v))
        self.assertTrue(np.all(0 <= self.model_class.w))
