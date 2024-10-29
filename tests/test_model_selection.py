import unittest

import numpy as np
import pandas as pd
from tests.constants import PATH_TO_GT
from tests.fixtures import BaseTest
import yaml

from pgm.model_selection.cross_validation import CrossValidation
from pgm.model_selection.labeling import predict_label
from pgm.model_selection.main import cross_validation


class TestCrossValidationModels(BaseTest):
    def setUp(self):
        self.models = {}
        model_files = [
            "setting_DynCRep.yaml",
            "setting_MTCOV.yaml",
            "setting_ACD.yaml",
            "setting_CRep.yaml",
            "setting_JointCRep.yaml",
        ]
        for model_file in model_files:
            with open(PATH_TO_GT / model_file, "r", encoding="utf8") as file:
                model_name = model_file.split(".")[0].split("_")[1]
                self.models[model_name] = yaml.safe_load(file)

    def run_cv_and_check_results(self, model_name):
        # Load the model settings
        model = self.models[model_name]
        # Change the output folder to the current temporary folder
        model["parameters"]["out_folder"] = [self.folder]
        # Change the output file to the current temporary folder
        model["output_file"] = self.folder + model["output_file"]
        # Change the path of the ground truth file
        model["ground_truth_file"] = (
            PATH_TO_GT / self.models[model_name]["ground_truth_file"]
        )
        # Run the cross-validation
        cross_validation(model_name, model["parameters"], model["input_params"])
        # Load the generated and ground truth dataframes
        generated_df = pd.read_csv(model["output_file"])
        ground_truth_df = pd.read_csv(model["ground_truth_file"])
        # Check that the generated dataframe is equal to the ground truth dataframe
        for column in generated_df.columns:
            for i in range(len(generated_df)):
                self.assertAlmostEqual(
                    generated_df[column][i],
                    ground_truth_df[column][i],
                    places=5,
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
        # Use the define_grid method to create a parameter grid
        param_grid = CrossValidation.define_grid(param1=[1, 2], param2=["a", "b"])

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
