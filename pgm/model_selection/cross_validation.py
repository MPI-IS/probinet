"""
Main function to implement cross-validation given a number of communities.

- Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
- Infer parameters on the training set;
- Calculate performance measures in the test set (AUC).
"""

from abc import ABC, abstractmethod
from itertools import product
import logging
from pathlib import Path
import pickle
import time

import numpy as np

from pgm.input.loader import import_data
from pgm.model_selection.masking import extract_mask_kfold, shuffle_indices_all_matrix

# TODO: optimize for big matrices (so when the input would be done with force_dense=False)


class CrossValidation(ABC):
    def __init__(
        self, algorithm, model_parameters, cv_parameters, numerical_parameters={}
    ):
        self.algorithm = algorithm
        for d in [model_parameters, cv_parameters, numerical_parameters]:
            for key, value in d.items():
                setattr(self, key, value)

    def prepare_output_directory(self):
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)

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
            print(f"Mask saved in: {outmask}")

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

    @abstractmethod
    def calculate_performance_and_prepare_comparison(self):
        """
        Calculate performance measures and prepare comparison.
        """

    @abstractmethod
    def save_results(self):
        """
        Save results in a csv file.
        """

    def run_single_iteration(self):
        """
        Run the cross-validation procedure.
        """

        # Prepare cv parameters
        self.prng = np.random.RandomState(seed=self.parameters["rseed"])
        self.rseed = self.prng.randint(1000)

        # Set up output directory
        self.prepare_output_directory()

        # Prepare parameters to load data
        adjacency = self.prepare_file_name()

        # Prepare output file
        if self.out_results:
            self.out_file = self.out_folder + adjacency + "_cv.csv"
            print(f"Results will be saved in: {self.out_file}")

        # Import data
        self.load_data()

        print("\n### CV procedure ###")
        time_start = time.time()

        # Prepare indices for cross-validation
        self.L = self.B.shape[0]
        self.N = self.B.shape[-1]
        self.indices = shuffle_indices_all_matrix(self.N, self.L, self.rseed)

        # Cross-validation loop
        for fold in range(self.NFold):
            print("\nFOLD ", fold)

            self.parameters["end_file"] = (
                self.end_file + "_" + str(fold) + "K" + str(self.parameters["K"])
            )
            # Extract mask for the current fold
            mask = self.extract_mask(fold)

            # Prepare and run the algorithm
            tic = time.time()
            outputs, algorithm_object = self.prepare_and_run(mask)

            # Output performance results
            self.calculate_performance_and_prepare_comparison(
                outputs, mask, fold, algorithm_object
            )

            print(f"Time elapsed: {np.round(time.time() - tic, 2)} seconds.")

            # Save results
            if self.out_results:
                self.save_results()

        print(f"\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.")

    def prepare_file_name(self):
        # Get the adjacency name
        adjacency_path = Path(self.adj)
        # Adjacency name
        adjacency = adjacency_path.stem
        # Adjacency suffix
        suffix = adjacency_path.suffix
        # Check if the adjacency is a csv or dat file
        if suffix not in [".dat", ".csv"]:
            logging.warning("Adjacency name not recognized.")

        return adjacency

    def run_cross_validation(self, **kwargs):
        """
        Run the cross-validation procedure over a grid of parameters.
        """
        # Define the grid of parameters
        param_grid = self.define_grid(**kwargs)
        # Loop over the grid of parameters
        for params in param_grid:
            for key, value in params.items():
                setattr(self, key, value)
            self.run_single_iteration()
