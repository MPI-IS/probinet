"""
This is the test module for the CRep algorithm.
"""

import importlib.resources as importlib_resources
import unittest
from pathlib import Path

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.crep import CRep


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        """

        # Test case parameters
        self.algorithm = 'CRep'
        self.K = 3
        self.in_folder = Path('pgm') / 'data' / 'input'
        self.out_folder = Path('outputs')
        self.end_file = '_test'
        self.adj = 'syn111.dat'
        self.ego = 'source'
        self.alter = 'target'
        self.force_dense = False
        self.flag_conv = 'log'

        # Import data

        network = self.in_folder / self.adj  # network complete path
        self.A, self.B, self.B_T, self.data_T_vals = import_data(network,
                                                                 ego=self.ego,
                                                                 alter=self.alter,
                                                                 force_dense=self.force_dense,
                                                                 header=0)
        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm

        with importlib_resources.open_binary('pgm.data.model', 'setting_' + self.algorithm + '.yaml') as fp:
            conf = yaml.load(fp, Loader=yaml.Loader)

        conf['out_folder'] = self.out_folder

        self.model = CRep(N=self.A[0].number_of_nodes(), L=len(self.A), K=self.K, **conf)

    # test case function to check the crep.set_name function
    def test_import_data(self):
        """
        Test import data function
        """

        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
        else:
            self.assertTrue(self.B.vals.sum() > 0)

    # test case function to check the Person.get_name function
    def test_running_algorithm(self):
        """
        Test running algorithm function.
        """

        _ = self.model.fit(data=self.B,
                           data_T=self.B_T,
                           data_T_vals=self.data_T_vals,
                           flag_conv=self.flag_conv,
                           nodes=self.nodes)
        theta = np.load((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix(
            '.npz'))
        thetaGT = np.load((self.model.out_folder / str('theta_' + self.algorithm)).with_suffix(
            '.npz'))

        self.assertTrue(np.array_equal(self.model.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.model.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.model.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.model.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))
