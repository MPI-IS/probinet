"""
Main function to implement cross-validation given a number of communities.

- Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
- Infer parameters on the training set;
- Calculate performance measures in the test set (AUC).
"""

from abc import ABC, abstractmethod
import csv
from itertools import product
import logging
import os
from pathlib import Path
import pickle
import time

import numpy as np

from pgm.input.loader import import_data
from pgm.model_selection.masking import extract_mask_kfold, shuffle_indices_all_matrix
from pgm.output.evaluate import (
    calculate_AUC, calculate_conditional_expectation, calculate_expectation)
from pgm.output.likelihood import calculate_opt_func

# TODO: optimize for big matrices (so when the input would be done with force_dense=False)


class CrossValidation(ABC):
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
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

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

    @abstractmethod
    def load_data(self):
        """
        Load data from the input folder.
        """

    def _load_data(self):
        """
        Auxiliary method to load data from the input folder.
        """
        # Load data
        self.A, self.B, self.B_T, self.data_T_vals = import_data(
            self.in_folder + self.adj,
            ego=self.ego,
            alter=self.alter,
            force_dense=True,
            header=0,
        )
        # Get the nodes
        self.nodes = self.A[0].nodes()

    @abstractmethod
    def prepare_and_run(self):
        """
        Prepare and run the algorithm.
        """


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
        comparison["auc_train"] = calculate_AUC(M, self.B, mask=np.logical_not(mask))

        # Calculate the AUC for the test set (where mask is applied)
        comparison["auc_test"] = calculate_AUC(M, self.B, mask=mask)

        # Calculate the conditional expectation matrix M_cond
        M_cond = calculate_conditional_expectation(self.B, u, v, w, eta=eta)

        # Calculate the conditional AUC for the training set
        comparison["auc_cond_train"] = calculate_AUC(
            M_cond, self.B, mask=np.logical_not(mask)
        )

        # Calculate the conditional AUC for the test set
        comparison["auc_cond_test"] = calculate_AUC(M_cond, self.B, mask=mask)

        # Calculate the optimization function value for the training set
        comparison["opt_func_train"] = calculate_opt_func(
            self.B,
            algorithm_object,
            mask=mask,
            assortative=self.parameters["assortative"],
        )

        # Store the comparison list in the instance variable
        return comparison

    def save_results(self):
        # Check if the output file exists; if not, write the header
        output_path = Path(self.out_file)
        if not output_path.is_file():  # write header
            with output_path.open("w") as outfile:
                # Create a CSV writer object
                wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
                # Write the header row to the CSV file
                wrtr.writerow(list(self.comparison.keys()))

        # Open the output file in append mode
        with output_path.open("a") as outfile:
            # Create a CSV writer object
            wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
            # Write the comparison data to the CSV file
            wrtr.writerow(list(self.comparison.values()))
            # Flush the output buffer to ensure all data is written to the file
            outfile.flush()

    def run_single_iteration(self):
        """
        Run the cross-validation procedure.
        """
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

        # Prepare cv parameters
        self.prng = np.random.RandomState(seed=self.parameters["rseed"])
        self.rseed = self.prng.randint(1000)

        # Set up output directory
        self.prepare_output_directory()

        # Prepare list to store results
        self.comparison = []

        # Prepare parameters to load data
        adjacency = self.prepare_file_name()

        # Prepare output file
        if self.out_results:
            self.out_file = self.out_folder + adjacency + "_cv.csv"
            logging.info("Results will be saved in: %s" % self.out_file)

        # Import data
        self.load_data()

        logging.info("Starting the cross-validation procedure.")
        time_start = time.time()

        # Prepare indices for cross-validation
        self.L = self.B.shape[0]
        self.N = self.B.shape[-1]
        self.indices = shuffle_indices_all_matrix(self.N, self.L, self.rseed)

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
            self.comparison.append(self.calculate_performance_and_prepare_comparison(
                outputs, mask, fold, algorithm_object
            ))

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

    # def run_cross_validation(self, **kwargs):
    #     """
    #     Run the cross-validation procedure over a grid of parameters.
    #     """
    #     # Define the grid of parameters
    #     param_grid = self.define_grid(**kwargs)
    #     # Loop over the grid of parameters
    #     for params in param_grid:
    #         for key, value in params.items():
    #             setattr(self, key, value)
    #         self.run_single_iteration()
    #     # checking that it exists
    #     print()
