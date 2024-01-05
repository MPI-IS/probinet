from pathlib import Path
import shutil
import sys
from unittest import mock, TestCase

import networkx as nx
import yaml

from pgm.main_CRep import main


class TestCRepMain(TestCase):

    def setUp(self):
        # Create a temporary output folder for testing
        self.temp_output_folder = Path('./temp_CRep_output/')
        self.temp_output_folder.mkdir()

    def tearDown(self):
        # Remove the temporary output folder after the test
        shutil.rmtree(self.temp_output_folder)

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
            'end_file': '_test',
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

        # Check if the fit method is called with the correct value of K
        mock_fit.assert_called_with(
            data=mock.ANY,
            data_T=mock.ANY,
            data_T_vals=mock.ANY,
            nodes=mock.ANY,
            K=expected_config['K'],  # Check if K is passed as an argument
            assortative=expected_config['assortative'],
            constrained=expected_config['constrained'],
            end_file=expected_config['end_file'],
            eta0=expected_config['eta0'],
            files=expected_config['files'],
            fix_eta=expected_config['fix_eta'],
            initialization=expected_config['initialization'],
            out_folder=expected_config['out_folder'],
            out_inference=expected_config['out_inference'],
            rseed=expected_config['rseed'],
            undirected=expected_config['undirected']
        )

    @mock.patch('pgm.model.crep.CRep.fit')
    @mock.patch('pgm.main_CRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):
        # Simulate running the script with custom parameters
        sys.argv = ['main_CRep.py', '-a', 'CRep', '-o', str(self.temp_output_folder),
                    '-K', '5', '-F', 'deltas', '-A', 'custom_network.dat']

        # Call the main function
        main()

        # Check that the import_data method is called
        mock_import_data.assert_called_once()

        # Check if the fit method is called with the correct values
        mock_fit.assert_called_with(
            data=mock.ANY,
            data_T=mock.ANY,
            data_T_vals=mock.ANY,
            nodes=mock.ANY,
            K=5,  # Custom value for K
            assortative=mock.ANY,  # Assuming assortative is included in sys.argv
            constrained=mock.ANY,  # Assuming constrained is included in sys.argv
            end_file=mock.ANY,  # Assuming end_file is included in sys.argv
            eta0=mock.ANY,  # Assuming eta0 is included in sys.argv
            files=mock.ANY,
            fix_eta=mock.ANY,
            initialization=mock.ANY,
            out_folder=mock.ANY,
            out_inference=mock.ANY,
            rseed=mock.ANY,
            undirected=mock.ANY
        )
