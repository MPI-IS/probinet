"""
Main function to implement cross-validation given a number of communities.

- Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
- Infer parameters on the training set;
- Calculate performance measures in the test set (AUC).
"""

import csv
import logging
import pickle
import time
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path

import numpy as np

from probinet.evaluation.expectation_computation import (
    calculate_conditional_expectation,
    calculate_expectation,
)

from ..evaluation.likelihood import calculate_opt_func
from ..evaluation.link_prediction import compute_link_prediction_AUC
from ..input.loader import build_adjacency_from_file
from ..models.classes import GraphData
from .masking import extract_mask_kfold, shuffle_indices_all_matrix


class CrossValidation(ABC):
    """
    Abstract class to implement cross-validation for a given algorithm.
    """

    def __init__(
        self, algorithm, model_parameters, cv_parameters, numerical_parameters=None
    ):
        self.algorithm = algorithm
        for key, value in model_parameters.items():
            setattr(self, key, value)
        for key, value in cv_parameters.items():
            setattr(self, key, value)
        for key, value in numerical_parameters.items():
            setattr(self, key, value)

    def prepare_output_directory(self):
        """
        Prepare the output directory to save the results.
        """
        output_path = Path(self.out_folder)
        output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_mask(self, fold):
        """
        Extract the mask for the current fold.
        """

    def _extract_mask(self, fold):
        """
        Auxiliary method to extract the mask for the current fold using k-fold cross-validation.
        """
        mask = extract_mask_kfold(self.indices, self.N, fold=fold, NFold=self.NFold)

        if self.out_mask:
            outmask = self.out_folder + "mask_f" + str(fold) + "_" + self.adj + ".pkl"
            logging.debug("Mask saved in: %s", outmask)

            with open(outmask, "wb") as f:
                pickle.dump(np.where(mask > 0), f)

        return mask

    @staticmethod
    def define_grid(**kwargs):
        """
        Define the grid of parameters to be tested.
        """
        # Get the parameter names and their corresponding values
        param_names = kwargs.keys()
        param_values = kwargs.values()

        # Create the Cartesian product of the parameter values
        param_combinations = product(*param_values)

        # Convert the Cartesian product into a list of dictionaries
        grid = [
            dict(zip(param_names, combination)) for combination in param_combinations
        ]

        return grid

    def load_data(self):
        """
        Auxiliary method to load data from the input folder.
        """
        # Load data
        self.gdata: GraphData = build_adjacency_from_file(
            self.in_folder + self.adj,
            ego=self.ego,
            alter=self.alter,
            force_dense=True,
            header=0,
            sep=self.sep,
        )

    def prepare_and_run(self, mask: np.ndarray):
        """
        Prepare the data for training and run the algorithm.

        Parameters
        ----------
        mask: np.ndarray
            The mask to apply on the data.

        Returns
        -------
        tuple
            The outputs of the algorithm.
        object
            The algorithm object.

        """
        # Create a copy of the adjacency matrix B to use for training
        B_train = self.gdata.adjacency_tensor.copy()

        # Apply the mask to the training data by setting masked elements to 0
        B_train[mask > 0] = 0

        # Create a copy of gdata to use for training
        self.gdata_for_training = self.gdata._replace(adjacency_tensor=B_train)

        # Initialize the algorithm object
        algorithm_object = self.model(**self.num_parameters)

        # Define rng parameter from the previously defined rng
        self.parameters["rng"] = self.rng

        # Fit the CRep models to the training data and get the outputs
        outputs = algorithm_object.fit(self.gdata_for_training, **self.parameters)

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def _calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object
    ):
        # Unpack the outputs from the algorithm
        u, v, w, eta, maxL = outputs

        # Initialize the comparison dictionary with keys as headers
        comparison = {
            "K": self.parameters["K"],
            "fold": fold,
            "rseed": self.parameters["rseed"],
            "eta": eta,
            "maxL": maxL,
            "final_it": algorithm_object.final_it,
        }

        # Calculate the expected matrix M using the parameters u, v, w, and eta
        M = calculate_expectation(u, v, w, eta=eta)

        # Calculate the AUC for the training set (where mask is not applied)
        comparison["auc_train"] = compute_link_prediction_AUC(
            self.gdata.adjacency_tensor, M, mask=np.logical_not(mask)
        )

        # Calculate the AUC for the test set (where mask is applied)
        comparison["auc_test"] = compute_link_prediction_AUC(
            self.gdata.adjacency_tensor, M, mask=mask
        )

        # Calculate the conditional expectation matrix M_cond
        M_cond = calculate_conditional_expectation(
            self.gdata.adjacency_tensor, u, v, w, eta=eta
        )

        # Calculate the conditional AUC for the training set
        comparison["auc_cond_train"] = compute_link_prediction_AUC(
            M_cond, self.gdata.adjacency_tensor, mask=np.logical_not(mask)
        )

        # Calculate the conditional AUC for the test set
        comparison["auc_cond_test"] = compute_link_prediction_AUC(
            M_cond, self.gdata.adjacency_tensor, mask=mask
        )

        # Calculate the optimization function value for the training set
        comparison["opt_func_train"] = calculate_opt_func(
            self.gdata.adjacency_tensor,
            algorithm_object,
            mask=mask,
            assortative=self.parameters["assortative"],
        )

        # Store the comparison list in the instance variable
        return comparison

    def save_results(self):
        # Check if the evaluation file exists; if not, write the header
        output_path = Path(self.out_file)
        if not output_path.is_file():  # write header
            with output_path.open("w", encoding="utf-8") as outfile:
                # Create a CSV writer object
                wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
                # Write the header row to the CSV file
                wrtr.writerow(list(self.comparison.keys()))

        # Open the evaluation file in append mode
        with output_path.open("a", encoding="utf-8") as outfile:
            # Create a CSV writer object
            wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
            # Write the comparison data to the CSV file
            wrtr.writerow(list(self.comparison.values()))
            # Flush the evaluation buffer to ensure all data is written to the file
            outfile.flush()

    def run_single_iteration(self):
        """
        Run the cross-validation procedure.
        """
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

        # Prepare cv parameters
        self.rng = np.random.default_rng(seed=self.parameters["rseed"])

        # Set up evaluation directory
        self.prepare_output_directory()

        # Prepare list to store results
        self.comparison = []

        # Prepare parameters to load data
        adjacency = self.prepare_file_name()

        # Prepare evaluation file
        if self.out_results:
            self.out_file = self.out_folder + adjacency + "_cv.csv"
            logging.info("Results will be saved in: %s" % self.out_file)

        # Import data
        self.load_data()

        logging.info("Starting the cross-validation procedure.")
        time_start = time.time()

        # Prepare indices for cross-validation
        self.L = self.gdata.adjacency_tensor.shape[0]
        self.N = self.gdata.adjacency_tensor.shape[-1]
        self.indices = shuffle_indices_all_matrix(self.N, self.L, rng=self.rng)

        # Cross-validation loop
        for fold in range(self.NFold):
            logging.info("\nFOLD %s" % fold)

            self.parameters["end_file"] = (
                self.end_file + "_" + str(fold) + "K" + str(self.parameters["K"])
            )
            # Extract mask for the current fold
            mask = self.extract_mask(fold)

            # Prepare and run the algorithm
            tic = time.time()
            outputs, algorithm_object = self.prepare_and_run(mask)

            # Output performance results
            self.comparison.append(
                self.calculate_performance_and_prepare_comparison(
                    outputs, mask, fold, algorithm_object
                )
            )

            logging.info("Time elapsed: %s seconds." % np.round(time.time() - tic, 2))
            logging.info(
                "\nTime elapsed: %s seconds." % np.round(time.time() - time_start, 2)
            )

        return self.comparison

    def prepare_file_name(self):
        if ".dat" in self.adj:
            adjacency = self.adj.split(".dat")[0]
        elif ".csv" in self.adj:
            adjacency = self.adj.split(".csv")[0]
        else:
            adjacency = self.adj
            # Warning: adjacency is not csv nor dat
            logging.warning("Adjacency name not recognized.")
        return adjacency
