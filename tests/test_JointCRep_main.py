import sys
import tempfile
from unittest import mock, TestCase

import networkx as nx
import yaml

from pgm.main_JointCRep import main


class TestJointCRepMain(
        TestCase):  # TODO: Refactor this test by creating a function that runs the main
    # function with a given set of arguments and checks the output

    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = temp_output_folder
            # Call the parent class's run method to execute the test
            super().run(result)

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    def test_main_with_no_parameters(self, mock_fit):
        # Simulate running the script with command-line arguments
        sys.argv = ['main_JointCRep.py', '-a', 'JointCRep', '-o', self.temp_output_folder]

        # Call the main function
        main()

        # Check that the fit method is called
        mock_fit.assert_called_once()

        # Check the contents of the generated configuration file
        expected_config = {
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

        # Path to the generated configuration file
        config_file_path = self.temp_output_folder + '/setting_JointCRep.yaml'

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

    @mock.patch('pgm.model.jointcrep.JointCRep.fit')
    @mock.patch('pgm.main_JointCRep.import_data',
                return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY))
    def test_main_custom_parameters(self, mock_import_data, mock_fit):

        K = 5
        # Simulate running the script with custom parameters
        sys.argv = ['main_JointCRep.py', '-a', 'JointCRep', '-o', self.temp_output_folder,
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
        # Get the arguments passed to the mock
        called_args = mock_fit.call_args

        # Check that the right input names are passed to the fit method
        assert set(called_args.kwargs.keys()) == set(input_names)

        #  K has correct value too
        called_args.kwargs['K'] = K
