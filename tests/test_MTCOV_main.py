from pathlib import Path
import shutil
import sys
from unittest import mock, TestCase

import networkx as nx
import numpy as np
import yaml

from pgm.main_MTCOV import main


class TestMTCOVMain(TestCase):

    def setUp(self):
        # Create a temporary output folder for testing
        self.temp_output_folder = Path('./temp_MTCOV_output/')
        self.temp_output_folder.mkdir()

    def tearDown(self):
        # Remove the temporary output folder after the test
        shutil.rmtree(self.temp_output_folder)

    @mock.patch('pgm.model.mtcov.MTCOV.fit')
    def test_main_with_no_parameters(self, mock_fit):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_MTCOV.py', '-o', str(self.temp_output_folder)]

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
            "end_file": "_MTCOV",
            "assortative": False,
            "files": "../data/input/theta.npz",
        }

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder / 'setting_MTCOV.yaml'

        # Load the actual configuration from the file
        with open(config_file_path, 'r', encoding='utf8') as f:
            actual_config = yaml.safe_load(f)

        # Compare the actual and expected configurations
        self.assertEqual(actual_config, expected_config)

        # Check if the fit method is called with the correct value of K
        mock_fit.assert_called_with(
            data=mock.ANY,
            data_X=mock.ANY,
            nodes=mock.ANY,
            flag_conv='log',
            K=expected_config['K'],  # Check if K is passed as an argument
            assortative=expected_config['assortative'],
            end_file=expected_config['end_file'],
            files=expected_config['files'],
            initialization=expected_config['initialization'],
            out_folder=expected_config['out_folder'],
            out_inference=expected_config['out_inference'],
            rseed=expected_config['rseed'],
            batch_size=mock.ANY
        )

    @mock.patch('pgm.model.mtcov.MTCOV.fit')
    @mock.patch('pgm.main_MTCOV.import_data_mtcov',
                return_value=([nx.Graph()], np.empty(0), mock.ANY, list()))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):
        K = 5
        # Simulate running the script with custom parameters
        sys.argv = ['main_MTCOV.py', '-o', str(self.temp_output_folder),
                    '-K', str(K), '-F', 'deltas', '-j', 'custom_adj.csv']

        # Call the main function
        main()

        # Check that the import_data method is called
        mock_import_data.assert_called_once()

        # Check if the fit method is called with the correct values
        mock_fit.assert_called_with(
            data=mock.ANY,
            data_X=mock.ANY,
            flag_conv=mock.ANY,
            nodes=mock.ANY,
            batch_size=mock.ANY,
            K=K,  # Check if K is passed as an argument
            rseed=mock.ANY,
            initialization=mock.ANY,
            out_inference=mock.ANY,
            out_folder=mock.ANY,
            end_file=mock.ANY,
            assortative=mock.ANY,
            files=mock.ANY,
        )
