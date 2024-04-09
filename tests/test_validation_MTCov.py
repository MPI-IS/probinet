from importlib.resources import files
from pathlib import Path

import numpy as np
from tests.fixtures import BaseTest
import yaml

from pgm.input.loader import import_data_mtcov
from pgm.model.mtcov import MTCov


class MTCovTestCase(BaseTest):

    def setUp(self):
        """
        Set up the test case.
        """
        self.algorithm = 'MTCov'
        self.C = 2
        self.gamma = 0.5
        self.out_folder = 'outputs/'
        self.end_file = '_test'
        self.adj_name = 'adj.csv'
        self.cov_name = 'X.csv'
        self.ego = 'source'
        self.alter = 'target'
        self.egoX = 'Name'
        self.attr_name = 'Metadata'
        self.undirected = False
        self.flag_conv = 'log'
        self.force_dense = False
        self.batch_size = None

        # Import data
        pgm_data_input_path = 'pgm.data.input'
        self.A, self.B, self.X, self.nodes = import_data_mtcov(pgm_data_input_path,
                                                               adj_name=self.adj_name,
                                                               cov_name=self.cov_name,
                                                               ego=self.ego,
                                                               alter=self.alter,
                                                               egoX=self.egoX,
                                                               attr_name=self.attr_name,
                                                               undirected=self.undirected,
                                                               force_dense=self.force_dense)

        self.Xs = np.array(self.X)

        with (files('pgm.data.model').joinpath('setting_' + self.algorithm + '.yaml').open('rb')
              as fp):
            self.conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        self.conf['out_folder'] = self.folder

        self.conf['end_file'] = '_OUT_' + self.algorithm  # Adding a suffix to the output files

        self.model = MTCov()

    def test_import_data(self):
        # Check if the force_dense flag is set to True
        if self.force_dense:
            # If force_dense is True, assert that the sum of all elements in the matrix B is greater than 0
            self.assertTrue(self.B.sum() > 0)
        else:
            # If force_dense is False, assert that the sum of all values in the sparse matrix B is greater than 0
            self.assertTrue(self.B.vals.sum() > 0)

    def test_running_algorithm(self):


        _ = self.model.fit(data=self.B,
                           data_X=self.Xs,
                           flag_conv=self.flag_conv,
                           nodes=self.nodes,
                           batch_size=self.batch_size,
                           **self.conf)

        theta = np.load((Path(self.model.out_folder) / str('theta' +
                                                         self.model.end_file)).with_suffix(
            '.npz'))

        # This reads the synthetic data Ground Truth output
        thetaGT_path = Path(__file__).parent / 'outputs' / 'theta_GT_MTCov'
        thetaGT = np.load(thetaGT_path.with_suffix('.npz'))

        # Asserting the model information

        # Assert that the model's u_f attribute is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.u_f, theta['u']))

        # Assert that the model's v_f attribute is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.v_f, theta['v']))

        # Assert that the model's w_f attribute is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.w_f, theta['w']))

        # Assert that the model's beta_f attribute is close to the 'beta' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.beta_f, theta['beta']))

        # Asserting GT information

        # Assert that the 'u' value in the thetaGT dictionary is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['u'], theta['u']))

        # Assert that the 'v' value in the thetaGT dictionary is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['v'], theta['v']))

        # Assert that the 'w' value in the thetaGT dictionary is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['w'], theta['w']))

        # Assert that the 'beta' value in the thetaGT dictionary is close to the 'beta' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['beta'], theta['beta']))

