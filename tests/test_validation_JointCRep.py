"""
This is the test module for the JointCRep algorithm.
"""

from importlib.resources import open_binary
import os
from pathlib import Path
import unittest

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.jointcrep import JointCRep


class BaseTestCase(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Test case parameters
        self.algorithm = 'JointCRep'
        self.in_folder = Path('pgm') / 'data' / 'input'
        self.adj = 'synthetic_data.dat'  # TODO: Look for file
        self.ego = 'source'
        self.alter = 'target'
        self.K = 2  # TODO: Move to config file
        self.undirected = False  # TODO: Move to config file
        self.flag_conv = 'log'  # TODO: Move to config file
        self.force_dense = False  # TODO: Move to config file

        # Import data: removing self-loops and making binary

        network = self.in_folder / self.adj  # network complete path
        self.A, self.B, self.B_T, self.data_T_vals = import_data(
            network,
            ego=self.ego,
            alter=self.alter,
            undirected=self.undirected,
            force_dense=self.force_dense,
            noselfloop=True,
            verbose=True,
            binary=True,
            header=0
        )
        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm
        with open_binary('pgm.data.model', 'setting_' + self.algorithm + '.yaml') as fp:
            conf = yaml.load(fp, Loader=yaml.Loader)

        # Saving the outputs of the tests inside the tests dir
        conf['out_folder'] = Path(__file__).parent / conf['out_folder']

        conf['end_file'] = '_OUT_JointCRep'  # Adding a suffix to the output files

        self.conf = conf

        self.L = len(self.A)

        self.N = len(self.nodes)

        # Run model

        self.model = JointCRep(N=self.N, L=self.L, K=self.K, undirected=self.undirected, **conf)

    # test case function to check the JointCRep.set_name function
    def test_import_data(self):
        print("Start import data test\n")
        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
            print('B has ', self.B.sum(), ' total weight.')
        else:
            self.assertTrue(self.B.vals.sum() > 0)
            print('B has ', self.B.vals.sum(), ' total weight.')

    # test case function to check the JointCRep.get_name function
    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        _ = self.model.fit(data=self.B, data_T=self.B_T, data_T_vals=self.data_T_vals,
                           flag_conv=self.flag_conv, nodes=self.nodes)

        theta = np.load((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix(
            '.npz'))
        # This reads the synthetic data Ground Truth output
        thetaGT = np.load((self.model.out_folder / 'theta_GT_JointCRep').with_suffix('.npz'))

        self.assertTrue(np.array_equal(self.model.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.model.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.model.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.model.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))

        # Remove output npz files after testing using os module
        os.remove((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix('.npz'))
