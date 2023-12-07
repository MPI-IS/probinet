import importlib.resources as importlib_resources
import unittest
from pathlib import Path

import numpy as np
# import JointCRep
import yaml

from pgm.input.loader import import_data
from pgm.model.jointcrep import JointCRep


# import tools as tl


class Test(unittest.TestCase):
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

        '''
        Import data: removing self-loops and making binary
        '''
        network = self.in_folder / self.adj  # network complete path
        self.A, self.B, self.B_T, self.data_T_vals = import_data(network,
                                                                 ego=self.ego,
                                                                 alter=self.alter,
                                                                 undirected=self.undirected,
                                                                 force_dense=self.force_dense,
                                                                 noselfloop=True,
                                                                 verbose=True,
                                                                 binary=True,
                                                                 header=0)
        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm

        with importlib_resources.open_binary('pgm.data.model', 'setting_' + self.algorithm + '.yaml') as fp:
            conf = yaml.load(fp, Loader=yaml.Loader)

        # out_folder = '../data/output/'  # it seems that this is not needed
        conf['out_folder'] = 'tests/' + conf['out_folder']  # Saving the outputs of the tests inside the tests dir

        self.conf = conf

        self.L = len(self.A)

        self.N = len(self.nodes)

        # self.model = CRep()

        '''
        Run model
        '''
        self.model = JointCRep(N=self.N, L=self.L, K=self.K, undirected=self.undirected, **conf)
        # _ = mod_multicrep.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, flag_conv=flag_conv, nodes=nodes)

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

        # theta = np.load(self.model.out_folder + 'theta' + self.model.end_file + '.npz')
        theta = np.load((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix(
            '.npz'))
        thetaGT = np.load((self.model.out_folder / 'theta_synthetic_data').with_suffix('.npz'))

        self.assertTrue(np.array_equal(self.model.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.model.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.model.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.model.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))
