from pathlib import Path
import sys
import tempfile
from unittest import mock, TestCase

import networkx as nx
import numpy as np
import yaml

from pgm.main_MTCov import main


class TestMTCovMain(TestCase):

    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = Path(temp_output_folder)
            # Call the parent class's run method to execute the test
            super().run(result)

    @mock.patch('pgm.model.mtcov.MTCov.fit')
    def test_main_with_no_parameters(self, mock_fit):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_MTCov.py', '-o', str(self.temp_output_folder)]

        # Call the main function
        main()

        # Check that the fit method is called
        mock_fit.assert_called_once()

        expected_config = {
            "K": 2,
            "rseed": 107261,
            "initialization": 0,
            "out_inference": True,
            "out_folder": str(self.temp_output_folder),
            "end_file": "_MTCov",
            "assortative": False,
            "files": "../data/input/theta.npz",
        }

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder / 'setting_MTCov.yaml'

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
        for key in ['data', 'data_X', 'nodes']:
            self.assertIn(key, called_args.kwargs, f"{key} not found in called_kwargs")


    @mock.patch('pgm.model.mtcov.MTCov.fit')
    @mock.patch('pgm.main_MTCov.import_data_mtcov',
                return_value=([nx.Graph()], np.empty(0), mock.ANY, list()))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):
        K = 5
        # Simulate running the script with custom parameters
        sys.argv = ['main_MTCov.py', '-o', str(self.temp_output_folder),
                    '-K', str(K), '-F', 'deltas', '-j', 'custom_adj.csv']

        # Call the main function
        main()

        # Check that the import_data method is called
        mock_import_data.assert_called_once()

        input_names = [
            'data',
            'data_X',
            'flag_conv',
            'nodes',
            'batch_size',
            'K',
            'rseed',
            'initialization',
            'out_inference',
            'out_folder',
            'end_file',
            'assortative',
            'files'
        ]

        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check that the right input names are passed to the fit method
        assert set(called_args.kwargs.keys()) == set(input_names)

        #  K has correct value too
        called_args.kwargs['K'] = K
