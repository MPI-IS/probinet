from pathlib import Path
import sys
import tempfile
from unittest import mock, TestCase

import networkx as nx
import yaml

from pgm.main_CRep import main


class TestCRepMain(TestCase):
    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = temp_output_folder
            # Call the parent class's run method to execute the test
            super().run(result)

    @mock.patch('pgm.model.crep.CRep.fit')
    def test_main_with_no_parameters(self, mock_fit):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_CRep.py', '-a', 'CRep', '-o', str(self.temp_output_folder)]

        # Call the main function
        main()

        # Check that the fit method is called
        mock_fit.assert_called_once()

        # Check the contents of the generated configuration file
        expected_config = {
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

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder / 'setting_CRep.yaml'

        # Load the actual configuration from the file
        with open(config_file_path, 'r', encoding='utf8') as f:
            actual_config = yaml.safe_load(f)

        # Compare the actual and expected configurations
        self.assertEqual(actual_config, expected_config)

        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check if the fit method is called with the correct values
        for key in expected_config:
            self.assertEqual(called_args.kwargs[key], expected_config[key])

        # Check if specific keys are present in the kwargs
        for key in ['data', 'data_T', 'data_T_vals', 'nodes']:
            self.assertIn(key, called_args.kwargs, f"{key} not found in called_kwargs")

    @mock.patch('pgm.model.crep.CRep.fit')
    @mock.patch('pgm.main_CRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):

        K = 5
        # Simulate running the script with custom parameters
        sys.argv = ['main_CRep.py', '-a', 'CRep', '-o', str(self.temp_output_folder),
                    '-K', str(K), '-F', 'deltas', '-A', 'custom_network.dat']

        # Call the main function
        main()

        # Check that the import_data method is called
        mock_import_data.assert_called_once()

        input_names = [
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

        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check that the right input names are passed to the fit method
        assert set(called_args.kwargs.keys()) == set(input_names)

        #  K has correct value too
        called_args.kwargs['K'] = K
