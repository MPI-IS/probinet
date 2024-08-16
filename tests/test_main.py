import sys
from unittest import mock

import networkx as nx
import numpy as np
from tests.fixtures import BaseTest

from pgm.main import main as single_main


class TestMain(BaseTest):

    def setUp(self):
        self.expected_config = {}
        self.kwargs_to_check = {}
        self.input_names = {}
        self.K_values = {}

    def main_with_no_parameters(self, algorithm, mock_fit, main_function):

        # Set up the command line arguments for the main function
        sys.argv = [
            "main_" + algorithm,  # Name of the main script
            "-a", algorithm,  # Algorithm name
            "-o", str(self.folder),  # Output folder
            "-out_inference",  # Flag to output inference results
        ]

        # Call the main function
        main_function()

        # Ensure the fit method was called exactly once
        mock_fit.assert_called_once()

        # Get the arguments with which the fit method was called
        called_args = mock_fit.call_args

        # Check that each expected configuration key matches the called arguments
        for key in self.expected_config:
            self.assertEqual(called_args.kwargs[key], self.expected_config[key])

        # Check that each key in kwargs_to_check is present in the called arguments
        for key in self.kwargs_to_check:
            self.assertIn(key, called_args.kwargs, f"{key} not found in called_kwargs")

    def main_with_custom_parameters(
        self, algorithm, mock_import_data, mock_fit, main_function
    ):
        # Set the value of K from the class attribute
        K = self.K_values

        # Set up the command line arguments for the main function
        sys.argv = [
            "main_" + algorithm,  # Name of the main script
            "-a", algorithm,  # Algorithm name
            "-o", str(self.folder),  # Output folder
            "-K", str(K),  # Number of communities
            "-F", "deltas",  # Some parameter F with value 'deltas'
            "-A", "custom_network.dat",  # Some parameter A with value 'custom_network.dat'
            "--rseed", "0",  # Random seed
            "-out_inference",  # Flag to output inference results
        ]

        # Call the main function
        main_function()

        # Ensure the import_data function was called exactly once
        mock_import_data.assert_called_once()

        # Get the arguments with which the fit method was called
        called_args = mock_fit.call_args

        # Check that the keys in the called arguments match the expected input names
        assert set(called_args.kwargs.keys()) == set(self.input_names)

        # Check that the value of K in the called arguments matches the expected value
        assert called_args.kwargs["K"] == K


class TestMainCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 3,
            "assortative": False,
            "constrained": False,
            "end_file": "_CRep",
            "eta0": None,
            "files": "",
            "fix_eta": False,
            "initialization": 0,
            "out_folder": str(self.folder),
            "out_inference": True,
            "rseed": 0,
            "undirected": False,
            "mask": None
        }
        self.kwargs_to_check = ["data", "data_T", "data_T_vals", "nodes"]

        self.input_names = [
            "data",
            "data_T",
            "data_T_vals",
            "nodes",
            "K",
            "assortative",
            "constrained",
            "end_file",
            "eta0",
            "files",
            "fix_eta",
            "initialization",
            "out_folder",
            "out_inference",
            "rseed",
            "undirected",
            "mask",
        ]

        self.K_values = 5

    @mock.patch("pgm.model.crep.CRep.fit")
    def test_CRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters("CRep", mock_fit, single_main)

    @mock.patch("pgm.model.crep.CRep.fit")
    @mock.patch(
        "pgm.main.import_data",
        return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY),
    )
    def test_CRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            "CRep", mock_import_data, mock_fit, single_main
        )


class TestMainJointCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "assortative": False,
            "end_file": "_JointCRep",
            "eta0": None,
            "files": '',
            "fix_communities": False,
            "fix_eta": False,
            "fix_w": False,
            "initialization": 0,
            "out_folder": self.folder,
            "out_inference": True,
            "rseed": 0,
            "use_approximation": False,
            "undirected": False,
        }
        self.kwargs_to_check = ["data", "data_T", "data_T_vals", "nodes"]

        self.input_names = [
            "data",
            "data_T",
            "data_T_vals",
            "nodes",
            "K",
            "rseed",
            "initialization",
            "out_inference",
            "out_folder",
            "end_file",
            "assortative",
            "eta0",
            "fix_eta",
            "fix_communities",
            "fix_w",
            "use_approximation",
            "files",
            "undirected",
        ]

        self.K_values = 5

    @mock.patch("pgm.model.jointcrep.JointCRep.fit")
    def test_JointCRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters("JointCRep", mock_fit, single_main)

    @mock.patch("pgm.model.jointcrep.JointCRep.fit")
    @mock.patch(
        "pgm.main.import_data",
        return_value=([nx.Graph()], mock.ANY, mock.ANY, mock.ANY),
    )
    def test_JointCRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            "JointCRep", mock_import_data, mock_fit, single_main
        )


class TestMainMTCOV(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "gamma": 0.5,
            "rseed": 0,
            "initialization": 0,
            "out_inference": True,
            "out_folder": str(self.folder),
            "end_file": "_MTCOV",
            "assortative": False,
            "files": "",
            "undirected": False,
        }

        self.kwargs_to_check = ["data", "data_X", "nodes"]
        self.input_names = [
            "data",
            "data_X",
            "nodes",
            "K",
            "gamma",
            "assortative",
            "end_file",
            "files",
            "initialization",
            "out_folder",
            "out_inference",
            "rseed",
            "batch_size",
            "undirected",
        ]
        self.K_values = 5

    @mock.patch("pgm.model.mtcov.MTCOV.fit")
    def test_MTCOV_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters("MTCOV", mock_fit, single_main)

    @mock.patch("pgm.model.mtcov.MTCOV.fit")
    @mock.patch(
        "pgm.main.import_data_mtcov",
        return_value=([nx.Graph()], np.empty(0), mock.ANY, []),
    )
    def test_MTCOV_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            "MTCOV", mock_import_data, mock_fit, single_main
        )


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
            "files": "",
            "fix_beta": False,
            "fix_communities": False,
            "fix_eta": False,
            "fix_w": False,
            "initialization": 0,
            "out_folder": str(self.folder),
            "out_inference": True,
            "rseed": 0,
            "undirected": False,
            "mask": None,
        }

        self.kwargs_to_check = [
            "data",
            "T",
            "nodes",
            "flag_data_T",
            "ag",
            "bg",
            "temporal",
        ]

        self.input_names = [
            "K",
            "T",
            "ag",
            "assortative",
            "beta0",
            "bg",
            "constrained",
            "constraintU",
            "data",
            "end_file",
            "eta0",
            "files",
            "fix_beta",
            "fix_communities",
            "fix_eta",
            "fix_w",
            "flag_data_T",
            "initialization",
            "nodes",
            "out_folder",
            "out_inference",
            "rseed",
            "temporal",
            "undirected",
            "mask"
        ]
        self.K_values = 2

    @mock.patch("pgm.model.dyncrep.DynCRep.fit")
    def test_DynCRep_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters("DynCRep", mock_fit, single_main)

    @mock.patch("pgm.model.dyncrep.DynCRep.fit")
    @mock.patch(
        "pgm.main.import_data", return_value=([nx.Graph()], np.empty(0), mock.ANY, [])
    )
    def test_DynCRep_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            "DynCRep", mock_import_data, mock_fit, single_main
        )


class TestMainAnomalyDetection(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 3,
            "files": "",
            "fix_communities": False,
            "end_file": "_ACD",
            "out_folder": str(self.folder),
            "out_inference": True,
            "verbose": 0,
        }

        self.kwargs_to_check = [
            "data",
            "nodes",
            "undirected",
            "initialization",
            "assortative",
            "constrained",
            "ag",
            "bg",
            "pibr0",
            "mupr0",
            "end_file",
            "flag_anomaly",
            "fix_pibr",
            "fix_mupr",
            "K",
            "fix_communities",
            "files",
            "out_inference",
            "verbose",
            "out_folder",
        ]

        self.input_names = [
            "data",
            "nodes",
            "undirected",
            "initialization",
            "assortative",
            "constrained",
            "ag",
            "bg",
            "pibr0",
            "mupr0",
            "end_file",
            "flag_anomaly",
            "fix_pibr",
            "fix_mupr",
            "K",
            "fix_communities",
            "files",
            "out_inference",
            "verbose",
            "out_folder",
            "rseed",
        ]
        self.K_values = 2

    @mock.patch("pgm.model.acd.AnomalyDetection.fit")
    def test_ACD_with_no_parameters(self, mock_fit):
        return self.main_with_no_parameters("ACD", mock_fit, single_main)

    @mock.patch("pgm.model.acd.AnomalyDetection.fit")
    @mock.patch(
        "pgm.main.import_data", return_value=([nx.Graph()], np.empty(0), mock.ANY, [])
    )
    def test_ACD_with_custom_parameters(self, mock_import_data, mock_fit):
        return self.main_with_custom_parameters(
            "ACD", mock_import_data, mock_fit, single_main
        )
