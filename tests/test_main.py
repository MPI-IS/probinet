import sys
from pathlib import Path
from unittest import mock

from tests.fixtures import BaseTest

from probinet.main import main as single_main
from probinet.main import parse_args

COMMAND_TO_RUN_MODEL = "run_model"
PATH_TO_INPUT = str(Path("probinet") / "data" / "input")


class TestMain(BaseTest):
    def setUp(self):
        self.expected_config = {}
        self.kwargs_to_check = {}
        self.input_names = {}
        self.K_values = {}
        self.number_of_args_for_fit = 1

    def check_default_params_passed_to_fit(
        self, algorithm, mock_fit, main_function, *extra_args
    ):
        # Set up the command line arguments for the main function
        sys.argv = [
            COMMAND_TO_RUN_MODEL,  # Name of the main script
            algorithm,  # Algorithm name
            "--out_folder",
            str(self.folder),  # Output folder
            "--out_inference",  # Flag to evaluation inference results
            "--files",
            PATH_TO_INPUT,
            "--end_file",
            f"_{algorithm}",
        ] + list(
            extra_args
        )  # Add extra arguments if any

        # Call the main function
        main_function()

        # Ensure the fit method was called exactly once
        mock_fit.assert_called_once()

        # Get the arguments with which the fit method was called
        called_args = mock_fit.call_args

        # Check that the args are three
        self.assertEqual(len(called_args.args), self.number_of_args_for_fit)

        # Check that each expected configuration key and value is in the called arguments
        for key, value in self.expected_config.items():
            self.assertIn(
                key, called_args.kwargs, "Key '%s' not found in called arguments" % key
            )
            if (
                key != "rseed"
            ):  # Skip rseed as it is generated based on the current time
                self.assertEqual(
                    called_args.kwargs[key],
                    value,
                    "Value for key '%s' does not match the expected value" % key,
                )

    def check_custom_params_passed_to_fit(
        self, algorithm, mock_fit, main_function, *extra_args
    ):
        # Set the value of K from the class attribute
        K = self.K_values

        # Set up the command line arguments for the main function
        sys.argv = [
            COMMAND_TO_RUN_MODEL,  # Name of the main script
            algorithm,  # Algorithm name
            "--out_folder",
            str(self.folder),  # Output folder
            "--out_inference",  # Flag to evaluation inference results
            "--files",
            PATH_TO_INPUT,  # Input files
            "--end_file",
            f"_{algorithm}",  # End file
            "-K",
            str(K),  # Number of communities
            "--initialization",
            "0",
            "--assortative",
        ] + list(
            extra_args
        )  # Add extra arguments if any

        # Call the main function
        main_function()

        # Get the arguments with which the fit method was called
        called_args = mock_fit.call_args

        # Check that the value of K in the called arguments matches the expected value
        self.assertEqual(called_args.kwargs["K"], K)

        # Check that the evaluation folder in the called arguments matches the expected value
        self.assertEqual(called_args.kwargs["out_folder"], str(self.folder))

        # Check that out_inference is set to True
        self.assertTrue(called_args.kwargs["out_inference"])

        # Check that the initialization is set to 0
        self.assertEqual(called_args.kwargs["initialization"], 0)

        # Check that assortative is set to True
        self.assertTrue(called_args.kwargs["assortative"])


class TestMainCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 3,
            "assortative": False,
            "constrained": False,
            "end_file": "_CRep",
            "eta0": None,
            "files": PATH_TO_INPUT,
            "fix_eta": False,
            "initialization": 0,
            "out_folder": str(self.folder),
            "out_inference": True,
            "rseed": 0,
            "undirected": False,
        }

        self.K_values = 5

    @mock.patch("probinet.models.crep.CRep.fit")
    def test_CRep_with_no_parameters(self, mock_fit):
        return self.check_default_params_passed_to_fit("CRep", mock_fit, single_main)

    @mock.patch("probinet.models.crep.CRep.fit")
    def test_CRep_with_custom_parameters(self, mock_fit):
        return self.check_custom_params_passed_to_fit("CRep", mock_fit, single_main)


class TestMainJointCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "assortative": False,
            "end_file": "_JointCRep",
            "eta0": None,
            "files": PATH_TO_INPUT,
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

        self.K_values = 5

    @mock.patch("probinet.models.jointcrep.JointCRep.fit")
    def test_JointCRep_with_no_parameters(self, mock_fit):
        return self.check_default_params_passed_to_fit(
            "JointCRep", mock_fit, single_main
        )

    @mock.patch("probinet.models.jointcrep.JointCRep.fit")
    def test_JointCRep_with_custom_parameters(self, mock_fit):
        return self.check_custom_params_passed_to_fit(
            "JointCRep", mock_fit, single_main
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
            "files": PATH_TO_INPUT,
            "undirected": False,
            "batch_size": None,
        }

        self.K_values = 5

    @mock.patch("probinet.models.mtcov.MTCOV.fit")
    def test_MTCOV_with_no_parameters(self, mock_fit):
        return self.check_default_params_passed_to_fit("MTCOV", mock_fit, single_main)

    @mock.patch("probinet.models.mtcov.MTCOV.fit")
    def test_MTCOV_with_custom_parameters(self, mock_fit):
        return self.check_custom_params_passed_to_fit("MTCOV", mock_fit, single_main)


class TestMainDynCRep(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 2,
            "assortative": False,
            "beta0": 0.25,
            "ag": 1.1,
            "bg": 0.5,
            "constrained": False,
            "constraintU": False,
            "end_file": "_DynCRep",
            "eta0": None,
            "files": PATH_TO_INPUT,
            "fix_beta": False,
            "fix_communities": False,
            "fix_eta": False,
            "fix_w": False,
            "flag_data_T": 0,
            "initialization": 0,
            "out_folder": str(self.folder),
            "out_inference": True,
            "rseed": 0,
            "temporal": True,
            "undirected": False,
        }

        self.K_values = 2
        self.extra_args = ["--force_dense"]

    @mock.patch("probinet.models.dyncrep.DynCRep.fit")
    def test_DynCRep_with_no_parameters(self, mock_fit):
        return self.check_default_params_passed_to_fit(
            "DynCRep", mock_fit, single_main, *self.extra_args
        )

    @mock.patch("probinet.models.dyncrep.DynCRep.fit")
    def test_DynCRep_with_custom_parameters(self, mock_fit):
        return self.check_custom_params_passed_to_fit(
            "DynCRep", mock_fit, single_main, *self.extra_args
        )


class TestMainAnomalyDetection(TestMain):
    def setUp(self):
        super().setUp()
        self.expected_config = {
            "K": 3,
            "files": PATH_TO_INPUT,
            "ag": 1.1,
            "bg": 0.5,
            "constrained": False,
            "assortative": False,
            "fix_communities": False,
            "fix_mupr": False,
            "fix_pibr": False,
            "flag_anomaly": True,
            "initialization": 0,
            "end_file": "_ACD",
            "mupr0": None,
            "out_folder": str(self.folder),
            "out_inference": True,
            "pibr0": None,
            "rseed": 0,
            "undirected": False,
        }

        self.K_values = 2

    @mock.patch("probinet.models.acd.AnomalyDetection.fit")
    def test_ACD_with_no_parameters(self, mock_fit):
        return self.check_default_params_passed_to_fit("ACD", mock_fit, single_main)

    @mock.patch("probinet.models.acd.AnomalyDetection.fit")
    def test_ACD_with_custom_parameters(self, mock_fit):
        return self.check_custom_params_passed_to_fit("ACD", mock_fit, single_main)


class TestParseArgs(BaseTest):
    """
    Test cases for the parse_args function to ensure the correct subparser is called
    based on the first argument and that the arguments are parsed correctly.
    """

    K_value = 5

    def test_crep_subparser(self):
        """
        Test that the CRep subparser is called and the arguments are parsed correctly.
        """
        test_args = ["run_model", "CRep", "-K", str(self.K_value)]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_args()
            # Check that the algorithm is correctly set to CRep
            self.assertEqual(args.algorithm, "CRep")
            # Check that the K value is correctly parsed
            self.assertEqual(args.K, self.K_value)
            # Check that an argument specific to another parser (e.g., T for DynCRep) is not set
            self.assertFalse(hasattr(args, "T"))

    def test_jointcrep_subparser(self):
        """
        Test that the JointCRep subparser is called and the arguments are parsed correctly.
        """
        test_args = ["run_model", "JointCRep", "-K", str(self.K_value)]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_args()
            # Check that the algorithm is correctly set to JointCRep
            self.assertEqual(args.algorithm, "JointCRep")
            # Check that the K value is correctly parsed
            self.assertEqual(args.K, self.K_value)
            # Check that an argument specific to another parser (e.g., gamma for MTCOV) is not set
            self.assertFalse(hasattr(args, "gamma"))

    def test_mtcov_subparser(self):
        """
        Test that the MTCOV subparser is called and the arguments are parsed correctly.
        """
        test_args = ["run_model", "MTCOV", "-K", str(self.K_value)]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_args()
            # Check that the algorithm is correctly set to MTCOV
            self.assertEqual(args.algorithm, "MTCOV")
            # Check that the K value is correctly parsed
            self.assertEqual(args.K, self.K_value)
            # Check that an argument specific to another parser (e.g., beta0 for DynCRep) is not set
            self.assertFalse(hasattr(args, "beta0"))

    def test_dyncrep_subparser(self):
        """
        Test that the DynCRep subparser is called and the arguments are parsed correctly.
        """
        test_args = ["run_model", "DynCRep", "-K", str(self.K_value)]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_args()
            # Check that the algorithm is correctly set to DynCRep
            self.assertEqual(args.algorithm, "DynCRep")
            # Check that the K value is correctly parsed
            self.assertEqual(args.K, self.K_value)
            # Check that an argument specific to another parser (e.g., flag_anomaly for ACD) is not set
            self.assertFalse(hasattr(args, "flag_anomaly"))

    def test_acd_subparser(self):
        """
        Test that the ACD subparser is called and the arguments are parsed correctly.
        """
        test_args = ["run_model", "ACD", "-K", str(self.K_value)]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_args()
            # Check that the algorithm is correctly set to ACD
            self.assertEqual(args.algorithm, "ACD")
            # Check that the K value is correctly parsed
            self.assertEqual(args.K, self.K_value)
            # Check that an argument specific to another parser (e.g., cov_name for MTCOV) is not set
            self.assertFalse(hasattr(args, "cov_name"))


class TestDefaultMainCalls(BaseTest):
    """
    Test cases for the CLI interface. The goal is to ensure that the main function runs correctly with the default parameters.
    """

    def run_algorithm_test(self, algorithm, *extra_args):
        # Set up the command line arguments for the main function
        sys.argv = ["run_model", algorithm, "-f", PATH_TO_INPUT, "-d"] + list(
            extra_args
        )
        # Call the main function
        single_main()

    def test_main_crep(self):
        self.run_algorithm_test("CRep")

    def test_main_crep_assortative(self):
        self.run_algorithm_test("CRep", "--assortative", "--num_realizations", "1")

    def test_main_jointcrep(self):
        self.run_algorithm_test("JointCRep")

    def test_main_jointcrep_approximation(self):
        self.run_algorithm_test(
            "JointCRep",
            "--use_approximation",
            "True",
            "--num_realizations",
            "2",
            "--max_iter",
            "1000",
        )

    def test_main_mtcov(self):
        self.run_algorithm_test("MTCOV")

    def test_main_mtcov_assortative(self):
        self.run_algorithm_test("MTCOV", "--assortative", "--num_realizations", "1")

    def test_main_dyncrep(self):
        self.run_algorithm_test("DynCRep", "--force_dense")

    def test_main_dyncrep_assortative(self):
        self.run_algorithm_test(
            "DynCRep", "--assortative", "--force_dense", "--num_realizations", "1"
        )

    def test_main_acd(self):
        self.run_algorithm_test("ACD")

    def test_main_acd_assortative(self):
        self.run_algorithm_test("ACD", "--assortative")
