import sys
from unittest import mock

import networkx as nx
import numpy as np
from tests.fixtures import BaseTest
import yaml

from pgm.main import main as single_main


class TestMain(BaseTest):

    def setUp(self):
        self.expected_config = {}
        self.kwargs_to_check = {}
        self.input_names = {}
        self.K_values = {}

    def main_with_no_parameters(self, algorithm, mock_fit, main_function):
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.folder)]
        main_function()
        mock_fit.assert_called_once()
        config_file_path = self.folder + '/setting_' + algorithm + '.yaml'
        with open(config_file_path, 'r', encoding='utf8') as f:
            actual_config = yaml.safe_load(f)
        self.assertEqual(actual_config, self.expected_config)
        called_args = mock_fit.call_args
        for key in self.expected_config:
            self.assertEqual(called_args.kwargs[key], self.expected_config[key])
        for key in self.kwargs_to_check:
            self.assertIn(key, called_args.kwargs, f"{key} not found in called_kwargs")

    def main_with_custom_parameters(self, algorithm, mock_import_data, mock_fit, main_function):
        K = self.K_values
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.folder),
                    '-K', str(K), '-F', 'deltas', '-A', 'custom_network.dat', '--rseed', '0']
        main_function()
        mock_import_data.assert_called_once()
        called_args = mock_fit.call_args
        assert set(called_args.kwargs.keys()) == set(self.input_names)
        assert called_args.kwargs['K'] == K


class TestMainCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            'K': 3,
            'assortative': True,
            'constrained': True,
            'end_file': '_CRep',
            'eta0': None,
            'files': 'config/data/input/theta_gt111.npz',
            'fix_eta': False,
            'initialization': 0,
            'out_folder': str(self.folder),
            'out_inference': True,
            'rseed': 0,
            'undirected': False
        }
        self.kwargs_to_check = ['data', 'data_T', 'data_T_vals', 'nodes']

        self.input_names = [
            'data',
            'data_T',
            'data_T_vals',
            'nodes',
            'K',
            'assortative',
            'constrained',
            'end_file',
            'eta0',
            'files',
            'fix_eta',
            'initialization',
            'out_folder',
            'out_inference',
            'rseed',
            'undirected'
        ]

        self.K_values = 5

    @mock.patch('pgm.model.crep.CRep.fit')
    def test_CRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('CRep', mock_fit, single_main)

    @mock.patch('pgm.model.crep.CRep.fit')
    @mock.patch('pgm.main.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_CRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('CRep', mock_import_data, mock_fit, single_main)


class TestMainJointCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            'K': 2,
            'assortative': False,
            'end_file': '_JointCRep',
            'eta0': None,
            'files': '../data/input/theta.npz',
            'fix_communities': False,
            'fix_eta': False,
            'fix_w': False,
            'initialization': 0,
            'out_folder': self.folder,
            'out_inference': True,
            'rseed': 10,
            'use_approximation': False,
        }
        self.kwargs_to_check = ['data', 'data_T', 'data_T_vals', 'nodes']

        self.input_names = [
            'data',
            'data_T',
            'data_T_vals',
            'nodes',
            'K',
            'rseed',
            'initialization',
            'out_inference',
            'out_folder',
            'end_file',
            'assortative',
            'eta0',
            'fix_eta',
            'fix_communities',
            'fix_w',
            'use_approximation',
            'files'
        ]

        self.K_values = 5

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    def test_JointCRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('JointCRep', mock_fit, single_main)

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    @mock.patch('pgm.main.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_JointCRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            'JointCRep', mock_import_data, mock_fit, single_main)


class TestMainMTCOV(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "gamma": 0.5,
            "rseed": 107261,
            "initialization": 0,
            "out_inference": True,
            "out_folder": str(self.folder),
            "end_file": "_MTCOV",
            "assortative": False,
            "files": "../data/input/theta.npz",
        }

        self.kwargs_to_check = ['data', 'data_X', 'nodes', 'flag_conv']
        self.input_names = [
            'data',
            'data_X',
            'nodes',
            'flag_conv',
            'K',
            'gamma',
            'assortative',
            'end_file',
            'files',
            'initialization',
            'out_folder',
            'out_inference',
            'rseed',
            'batch_size'
        ]
        self.K_values = 5

    @mock.patch('pgm.model.mtcov.MTCOV.fit')
    def test_MTCOV_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('MTCOV', mock_fit, single_main)

    @mock.patch('pgm.model.mtcov.MTCOV.fit')
    @mock.patch('pgm.main.import_data_mtcov',
                return_value=([nx.Graph()], np.empty(0), mock.ANY, []))
    def test_MTCOV_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('MTCOV', mock_import_data, mock_fit, single_main)


class TestMainDynCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "assortative": False,
            "beta0": 0.25,
            "constrained": False,
            "constraintU": False,
            "end_file": "_DynCRep",
            "eta0": None,
            "files": "tests/inputs/theta_GT_DynCRep_for_initialization.npz",
            "fix_beta": False,
            "fix_communities": False,
            "fix_eta": False,
            "fix_w": False,
            "initialization": 0,
            "out_folder": str(self.folder),
            "out_inference": True,
            "rseed": 0,
            "undirected": False
        }

        self.kwargs_to_check = ['data', 'T', 'nodes', 'flag_data_T', 'ag', 'bg', 'temporal']

        self.input_names = [
            'K',
            'T',
            'ag',
            'assortative',
            'beta0',
            'bg',
            'constrained',
            'constraintU',
            'data',
            'end_file',
            'eta0',
            'files',
            'fix_beta',
            'fix_communities',
            'fix_eta',
            'fix_w',
            'flag_data_T',
            'initialization',
            'nodes',
            'out_folder',
            'out_inference',
            'rseed',
            'temporal',
            'undirected'
        ]
        self.K_values = 2

    @mock.patch('pgm.model.dyncrep.DynCRep.fit')
    def test_DynCRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('DynCRep', mock_fit, single_main)

    @mock.patch('pgm.model.dyncrep.DynCRep.fit')
    @ mock.patch('pgm.main.import_data',
                 return_value=([nx.Graph()], np.empty(0), mock.ANY, []))
    def test_JointCRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            'DynCRep', mock_import_data, mock_fit, single_main)
