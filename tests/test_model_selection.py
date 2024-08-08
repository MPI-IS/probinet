import unittest

import numpy as np
import pandas as pd
from tests.constants import PATH_TO_GT
from tests.fixtures import BaseTest, ConcreteCrossValidation

from pgm.model_selection.labeling import predict_label
from pgm.model_selection.main import cross_validation


class TestCrossValidationModels(BaseTest):
    def setUp(self):
        self.models = {
            "DynCRep": {
                "parameters": {
                    "K": [4],
                    "T": [5],
                    "ag": [1.1],
                    "bg": [0.5],
                    "eta0": [0.2],
                    "flag_data_T": [0],
                    "beta0": [0.2],
                    "rseed": [100],
                    "fix_eta": [False],
                    "fix_beta": [False],
                    "fix_communities": [False],
                    "fix_w": [False],
                    "assortative": [False],
                    "constrained": [False],
                    "constraintU": [False],
                    "undirected": [False],
                    "temporal": [True],
                    "initialization": [0],
                    "out_inference": [False],
                    "out_folder": [self.folder],
                    "end_file": ["_DynCRep"],
                },
                "input_params": {
                    "in_folder": "pgm/data/input/",
                    "adj": "email-Eu-core.csv",
                    "ego": "source",
                    "alter": "target",
                    "sep": ",",
                    "NFold": 2,
                    "out_results": True,
                    "out_mask": False,
                },
                "output_file": self.folder + "email-Eu-core_cv.csv",
                "ground_truth_file": PATH_TO_GT + "email-Eu-core_cv_GT_DynCRep" ".csv",
            },
            "MTCOV": {
                "parameters": {
                    "K": [2],
                    "gamma": [0.5, 0.75],
                    "rseed": [107261],
                    "initialization": [0],
                    "out_inference": [False],
                    "out_folder": [self.folder],
                    "end_file": ["_MTCOV"],
                    "assortative": [False],
                    "undirected": [True],
                    "flag_conv": ["log"],
                    "batch_size": [None],
                    "files": ["../data/input/theta.npz"],
                },
                "input_params": {
                    "in_folder": "pgm/data/input/",
                    "adj": "adj_cv.csv",
                    "cov": "X_cv.csv",
                    "ego": "source",
                    "alter": "target",
                    "egoX": "Name",
                    "attr_name": "Metadata",
                    "NFold": 5,
                    "out_results": True,
                    "out_mask": False,
                },
                "output_file": self.folder + "adj_cv_cv.csv",
                "ground_truth_file": PATH_TO_GT + "/adj_cv_GT_MTCOV.csv",
            },
            "ACD": {
                "parameters": {
                    "K": [3],
                    "flag_anomaly": [True],
                    "rseed": [10],
                    "initialization": [0],
                    "out_inference": [False],
                    "out_folder": [self.folder + "2-fold_cv/"],
                    "end_file": ["_ACD"],
                    "assortative": [True],
                    "undirected": [False],
                    "files": ["../data/input/theta.npz"],
                },
                "input_params": {
                    "in_folder": "pgm/data/input/",
                    "adj": "synthetic_data_for_ACD.dat",
                    "ego": "source",
                    "alter": "target",
                    "NFold": 2,
                    "out_results": True,
                    "out_mask": False,
                },
                "output_file": self.folder + "2-fold_cv/synthetic_data_for_ACD_cv.csv",
                "ground_truth_file": PATH_TO_GT + "synthetic_data_for_ACD_cv_GT.csv",
            },
            "CRep": {
                "parameters": {
                    "K": [2, 3, 4],
                    "rseed": [623],
                    "initialization": [0],
                    "undirected": [False],
                    "end_file": ["_CRep"],
                    "assortative": [True],
                    "eta0": [None],
                    "fix_eta": [False],
                    "constrained": [True],
                    "out_inference": [False],
                    "out_folder": [self.folder],
                    "files": ["config/data/input/theta_gt111.npz"],
                },
                "input_params": {
                    "in_folder": "pgm/data/input/",
                    "adj": "syn111.dat",
                    "ego": "source",
                    "alter": "target",
                    "NFold": 2,
                    "out_results": True,
                    "out_mask": False,
                },
                "output_file": self.folder + "syn111_cv.csv",
                "ground_truth_file": PATH_TO_GT + "syn111_cv_GT_CRep.csv",
            },
            "JointCRep": {
                "parameters": {
                    "K": [2],
                    "rseed": [10],
                    "initialization": [0],
                    "out_inference": [False],
                    "out_folder": [self.folder],
                    "end_file": ["_JointCRep"],
                    "assortative": [False],
                    "eta0": [None],
                    "fix_eta": [False],
                    "undirected": [False],
                    "fix_communities": [False],
                    "fix_w": [False],
                    "use_approximation": [False],
                    "files": ["../data/input/theta.npz"],
                },
                "input_params": {
                    "in_folder": "pgm/data/input/",
                    "adj": "synthetic_data.dat",
                    "ego": "source",
                    "alter": "target",
                    "NFold": 2,
                    "out_results": True,
                    "out_mask": False,
                },
                "output_file": self.folder + "synthetic_data_cv.csv",
                "ground_truth_file": PATH_TO_GT + "synthetic_data_cv_GT_JointCRep.csv",
            },
        }

    def run_cv_and_check_results(self, model_name):
        model = self.models[model_name]
        cross_validation(model_name, model["parameters"], model["input_params"])

        generated_df = pd.read_csv(model["output_file"])
        ground_truth_df = pd.read_csv(model["ground_truth_file"])

        for column in generated_df.columns:
            pd.testing.assert_series_equal(
                generated_df[column],
                ground_truth_df[column],
                check_exact=False,
                rtol=1e-5,
            )

    def test_dyncrep_cross_validation(self):
        self.run_cv_and_check_results("DynCRep")

    def test_mtcov_cross_validation(self):
        self.run_cv_and_check_results("MTCOV")

    def test_acd_cross_validation(self):
        self.run_cv_and_check_results("ACD")

    def test_crep_cross_validation(self):
        self.run_cv_and_check_results("CRep")

    def test_jointcrep_cross_validation(self):
        self.run_cv_and_check_results("JointCRep")


class TestCrossValidation(unittest.TestCase):
    def test_define_grid(self):
        cv = ConcreteCrossValidation()
        param_grid = cv.define_grid(param1=[1, 2], param2=["a", "b"])

        expected_grid = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]

        self.assertEqual(param_grid, expected_grid)


class TestLabeling(unittest.TestCase):
    def test_predict_label_no_mask(self):
        # Create a sample design matrix X
        X = pd.DataFrame(
            {"label1": [1, 0, 0], "label2": [0, 1, 0], "label3": [0, 0, 1]}
        )

        # Create sample membership matrices u and v
        u = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        v = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])

        # Create a sample beta parameter matrix
        beta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Call the predict_label function
        predicted_labels = predict_label(X, u, v, beta)

        # Define the expected labels
        expected_labels = ["label1", "label2", "label3"]

        # Assert that the predicted labels match the expected labels
        self.assertEqual(predicted_labels, expected_labels)
