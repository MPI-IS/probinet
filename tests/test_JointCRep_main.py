from pathlib import Path
import shutil
import sys
from unittest import mock, TestCase

import networkx as nx
import yaml

from pgm.main_JointCRep import main


class TestJointCRepMain(TestCase):

    def setUp(self):
        # Create a temporary output folder for testing
        self.temp_output_folder = Path('./temp_JointCRep_output/')
        self.temp_output_folder.mkdir()

    def tearDown(self):
        # Remove the temporary output folder after the test
        shutil.rmtree(self.temp_output_folder)

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    def test_main_with_no_parameters(self, mock_fit):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_JointCRep.py', '-a', 'JointCRep', '-o', str(self.temp_output_folder)]

        # Call the main function
        main()

        # Check that the fit method is called
        mock_fit.assert_called_once()

        # Check the contents of the generated configuration file
        expected_config = {
            'K': 4,
            'assortative': False,
            'end_file': '_test',
            'eta0': None,
            'files': '../data/input/theta.npz',
            'fix_communities': False,
            'fix_eta': False,
            'fix_w': False,
            'initialization': 0,
            'num_realizations': 50,
            'out_folder': str(self.temp_output_folder),
            'out_inference': True,
            'plot_loglik': False,
            'rseed': 0,
            'use_approximation': False,
        }

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder / 'setting_JointCRep.yaml'

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
            end_file=expected_config['end_file'],
            eta0=expected_config['eta0'],
            files=expected_config['files'],
            fix_communities=expected_config['fix_communities'],
            fix_eta=expected_config['fix_eta'],
            fix_w=expected_config['fix_w'],
            initialization=expected_config['initialization'],
            num_realizations=expected_config['num_realizations'],
            out_folder=expected_config['out_folder'],
            out_inference=expected_config['out_inference'],
            plot_loglik=expected_config['plot_loglik'],
            rseed=expected_config['rseed'],
            use_approximation=expected_config['use_approximation']
        )

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    @mock.patch('pgm.main_JointCRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):
        K = 5
        # Simulate running the script with custom parameters
        sys.argv = ['main_JointCRep.py', '-a', 'JointCRep', '-o', str(self.temp_output_folder),
                    '-K', str(K), '-F', 'deltas', '-A', 'custom_network.dat']

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
            K=K,  # Check if K is passed as an argument
            rseed=mock.ANY,
            initialization=mock.ANY,
            out_inference=mock.ANY,
            out_folder=mock.ANY,
            end_file=mock.ANY,
            assortative=mock.ANY,
            eta0=mock.ANY,
            fix_eta=mock.ANY,
            fix_communities=mock.ANY,
            fix_w=mock.ANY,
            use_approximation=mock.ANY,
            files=mock.ANY,
            plot_loglik=mock.ANY,
            num_realizations=mock.ANY,
        )
