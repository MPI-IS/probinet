import sys
import tempfile
from unittest import mock, TestCase

import networkx as nx
import numpy as np
import yaml

from pgm.main_CRep import main as main_CRep_main
from pgm.main_JointCRep import main as main_JointCRep_main
from pgm.main_MTCov import main as main_MTCov_main


class TestMain(TestCase):
    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = temp_output_folder
            # Call the parent class's run method to execute the test
            super().run(result)

    def setUp(self):
        self.expected_config = {}
        self.expected_config['CRep'] = {
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
        self.expected_config['JointCRep'] = {
            'K': 4,
            'assortative': False,
            'end_file': '_JointCRep',
            'eta0': None,
            'files': '../data/input/theta.npz',
            'fix_communities': False,
            'fix_eta': False,
            'fix_w': False,
            'initialization': 0,
            'num_realizations': 50,
            'out_folder': self.temp_output_folder,
            'out_inference': True,
            'plot_loglik': False,
            'rseed': 0,
            'use_approximation': False,
        }
        self.expected_config['MTCov'] = {
            "K": 2,
            "rseed": 107261,
            "initialization": 0,
            "out_inference": True,
            "out_folder": str(self.temp_output_folder),
            "end_file": "_MTCov",
            "assortative": False,
            "files": "../data/input/theta.npz",
        }

        self.kwargs_to_check = {}
        self.kwargs_to_check['CRep'] = ['data', 'data_T', 'data_T_vals', 'nodes']
        self.kwargs_to_check['JointCRep'] = ['data', 'data_T', 'data_T_vals', 'nodes']
        self.kwargs_to_check['MTCov'] = ['data', 'data_X', 'nodes', 'flag_conv']

        self.input_names = {}
        self.input_names['CRep'] = [
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
        self.input_names['JointCRep'] = [
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
        self.input_names['MTCov'] = ['data', 'data_X', 'nodes', 'flag_conv', 'K', 'assortative', 'end_file', 'files',
                                     'initialization', 'out_folder', 'out_inference', 'rseed', 'batch_size']
        self.K_values = {}
        self.K_values['CRep'] = 5
        self.K_values['JointCRep'] = 5
        self.K_values['MTCov'] = 5

    def main_with_no_parameters(self, algorithm, mock_fit, main_function):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.temp_output_folder)]

        # If algorithm is MTCov,  then remove '-a' and the element after it. Why? Because the main function does not
        # have the flag '-a':
        if algorithm == 'MTCov':
            sys.argv.pop(1)  # Remove '-a'
            sys.argv.pop(1)  # Remove 'MTCov'

        # Call the main function
        main_function()

        # Check that the fit method is called
        mock_fit.assert_called_once()

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder + '/setting_' + algorithm + '.yaml'

        # Load the actual configuration from the file
        with open(config_file_path, 'r', encoding='utf8') as f:
            actual_config = yaml.safe_load(f)

        # Compare the actual and expected configurations
        self.assertEqual(actual_config, self.expected_config[algorithm])

        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check if the fit method is called with the correct values
        for key in self.expected_config[algorithm]:
            self.assertEqual(called_args.kwargs[key], self.expected_config[algorithm][key])

        # Check if specific keys are present in the kwargs
        for key in self.kwargs_to_check[algorithm]:
            self.assertIn(key, called_args.kwargs, f"{key} not found in called_kwargs")

    def main_with_custom_parameters(self, algorithm, mock_import_data, mock_fit, main_function):

        K = self.K_values[algorithm]
        # Simulate running the script with custom parameters
        sys.argv = ['main_' + algorithm, '-a', algorithm, '-o', str(self.temp_output_folder),
                    '-K', str(K)]
        if algorithm == 'CRep' or algorithm == 'JointCRep':
            sys.argv += ['-F', 'deltas', '-A', 'custom_network.dat']
        elif algorithm == 'MTCov':
            sys.argv += ['-F', 'deltas', '-j', 'custom_adj.csv']
            # Remove '-a' and the element after it. Why? Because the main function does not have the flag '-a'
            sys.argv.pop(1)  # Remove '-a'
            sys.argv.pop(1)  # Remove 'MTCov'

        # Call the main function
        main_function()

        # Check that the import_data method is called
        mock_import_data.assert_called_once()

        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check that the right input names are passed to the fit method
        assert set(called_args.kwargs.keys()) == set(self.input_names[algorithm])

        #  Check that K has correct value
        assert called_args.kwargs['K'] == K

    # Tests for CRep

    @mock.patch('pgm.model.crep.CRep.fit')
    def test_CRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('CRep', mock_fit, main_CRep_main)

    @mock.patch('pgm.model.crep.CRep.fit')
    @mock.patch('pgm.main_CRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_CRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('CRep', mock_import_data, mock_fit, main_CRep_main)

    # Tests for JointCRep

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    def test_JointCRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('JointCRep', mock_fit, main_JointCRep_main)

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    @mock.patch('pgm.main_JointCRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_JointCRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('JointCRep', mock_import_data, mock_fit, main_JointCRep_main)

    # Tests for MTCov
    @mock.patch('pgm.model.mtcov.MTCov.fit')
    def test_MTCov_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters('MTCov', mock_fit, main_MTCov_main)

    @mock.patch('pgm.model.mtcov.MTCov.fit')
    @mock.patch('pgm.main_MTCov.import_data_mtcov',
                return_value=([nx.Graph()], np.empty(0), mock.ANY, list()))
    def test_MTCov_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters('MTCov', mock_import_data, mock_fit, main_MTCov_main)
