import sys
import tempfile
from unittest import mock, TestCase

import networkx as nx
import numpy as np
import yaml

from pgm.main import main as single_main


class TestMain(TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as temp_output_folder:
            self.temp_output_folder = temp_output_folder
            super().run(result)

    def setUp(self):
        self.expected_config = {}
        self.kwargs_to_check = {}
        self.input_names = {}
        self.K_values = {}

    def main_with_no_parameters(self, algorithm, mock_fit, main_function):
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.temp_output_folder)]
        main_function()
        mock_fit.assert_called_once()
        config_file_path = self.temp_output_folder + '/setting_' + algorithm + '.yaml'
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
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.temp_output_folder),
                    '-K', str(K), '-F', 'deltas', '-A', 'custom_network.dat']
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
            'out_folder': str(self.temp_output_folder),
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
            'K': 4,
            'assortative': False,
            'end_file': '_JointCRep',
            'eta0': None,
            'files': '../data/input/theta.npz',
            'fix_communities': False,
            'fix_eta': False,
            'fix_w': False,
            'initialization': 0,
            'num_realizations': 3,
            'out_folder': self.temp_output_folder,
            'out_inference': True,
            'plot_loglik': False,
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
            'files',
            'plot_loglik',
            'num_realizations',
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


class TestMainMTCov(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "gamma": 0.5,
            "rseed": 107261,
            "initialization": 0,
            "out_inference": True,
            "out_folder": str(self.temp_output_folder),
            "end_file": "_MTCov",
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

    @mock.patch('pgm.model.mtcov.MTCov.fit')
    def test_MTCov_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('MTCov', mock_fit, single_main)

    @mock.patch('pgm.model.mtcov.MTCov.fit')
    @mock.patch('pgm.main.import_data_mtcov',
                return_value=([nx.Graph()], np.empty(0), mock.ANY, list()))
    def test_MTCOV_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('MTCov', mock_import_data, mock_fit, single_main)
