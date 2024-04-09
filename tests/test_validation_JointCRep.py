"""
This is the test module for the JointCRep algorithm.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
from tests.fixtures import BaseTest
import yaml

from pgm.input.loader import import_data
from pgm.model.jointcrep import JointCRep

# pylint: disable=missing-function-docstring, too-many-locals, too-many-instance-attributes


class  JointCRepTestCase(BaseTest):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Test case parameters
        self.algorithm = 'JointCRep'
        self.adj = 'synthetic_data.dat'
        self.ego = 'source'
        self.alter = 'target'
        self.K = 2
        self.undirected = False
        self.flag_conv = 'log'
        self.force_dense = False

        # Import data: removing self-loops and making binary

        with (files('pgm.data.input').joinpath(self.adj).open('rb') as network):
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
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

        with (files('pgm.data.model').joinpath('setting_' + self.algorithm + '.yaml').open('rb')
              as fp):
            conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        conf['out_folder'] = self.folder

        conf['end_file'] = '_OUT_JointCRep'  # Adding a suffix to the output files

        self.conf = conf

        self.conf['K'] = self.K

        self.L = len(self.A)

        self.N = len(self.nodes)

        # Run model

        self.model = JointCRep()

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

        _ = self.model.fit(data=self.B,
                           data_T=self.B_T,
                           data_T_vals=self.data_T_vals,
                           nodes=self.nodes,
                           **self.conf)

        theta = np.load((Path(self.model.out_folder) / str('theta' +
                                                         self.model.end_file)).with_suffix(
            '.npz'))

        # This reads the synthetic data Ground Truth output
        thetaGT_path = Path(__file__).parent / 'outputs' / 'theta_GT_JointCRep'
        thetaGT = np.load(thetaGT_path.with_suffix('.npz'))

        # Asserting the model information

        # Assert that the model's u_f attribute is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.u_f, theta['u']))

        # Assert that the model's v_f attribute is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.v_f, theta['v']))

        # Assert that the model's w_f attribute is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.w_f, theta['w']))

        # Assert that the model's eta_f attribute is close to the 'eta' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.eta_f, theta['eta']))

        # Asserting GT information

        # Assert that the 'u' value in the thetaGT dictionary is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['u'], theta['u']))

        # Assert that the 'v' value in the thetaGT dictionary is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['v'], theta['v']))

        # Assert that the 'w' value in the thetaGT dictionary is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['w'], theta['w']))

        # Assert that the 'eta' value in the thetaGT dictionary is close to the 'eta' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['eta'], theta['eta']))