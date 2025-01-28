"""
This module contains the DynCRepCrossValidation class, which is used for cross-validation of the
DynCRep algorithm.
"""

import logging
import time

import numpy as np

from ..evaluation.expectation_computation import (
    calculate_conditional_expectation_dyncrep,
)
from ..evaluation.likelihood import likelihood_conditional
from ..evaluation.link_prediction import compute_link_prediction_AUC
from ..models.dyncrep import DynCRep
from .cross_validation import CrossValidation


class DynCRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the DynCRep algorithm.

    - Hold-out the data at the latest time snapshot (at time T);
    - Infer parameters on the observed data (data up to time T-1);
    - Calculate performance measures in the hidden set (AUC).
    """

    def __init__(
        self, algorithm, parameters, input_cv_params, numerical_parameters=None
    ):
        """
        Constructor for the DynCRepCrossValidation class.
        Parameters
        ----------
        algorithm
        parameters
        input_cv_params
        numerical_parameters
        """
        super().__init__(algorithm, parameters, input_cv_params, numerical_parameters)
        # These are the parameters for the DynCRep algorithm
        self.parameters = parameters
        self.num_parameters = numerical_parameters
        self.model = DynCRep

    def extract_mask(self, fold):
        pass

    def prepare_and_run(self, t):
        # Create the training data
        B_train = self.gdata.adjacency_tensor[
            :t
        ]  # use data up to time t-1 for training

        # Create a copy of gdata to use for training
        self.gdata_for_training = self.gdata._replace(adjacency_tensor=B_train)

        self.parameters["T"] = t
        # Initialize the algorithm object
        algorithm_object = self.model(**self.num_parameters)

        # Define rng from the seed and add it to the parameters
        self.parameters["rng"] = np.random.default_rng(seed=self.parameters["rseed"])

        # Fit the model to the training data
        outputs = algorithm_object.fit(self.gdata_for_training, **self.parameters)

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self,
        outputs,
        _mask,
        fold,
        algorithm_object,
    ):
        """
        Calculate performance results and prepare comparison.
        """
        # Unpack the outputs from the algorithm
        u, v, w, eta, beta, maxL = outputs

        # Initialize the comparison dictionary with keys as headers
        comparison = {
            "algo": "DynCRep_temporal" if self.parameters["temporal"] else "DynCRep",
            "constrained": self.parameters["constrained"],
            "flag_data_T": self.parameters["flag_data_T"],
            "rseed": self.parameters["rseed"],
            "K": self.parameters["K"],
            "eta0": self.parameters["eta0"],
            "beta0": self.parameters["beta0"],
            "T": fold,
            "eta": eta,
            "beta": beta,
            "final_it": algorithm_object.final_it,
            "maxL": maxL,
        }

        if self.flag_data_T == 1:  # if 0: previous time step, 1: same time step
            M = calculate_conditional_expectation_dyncrep(
                self.B[fold], u, v, w, eta=eta, beta=beta
            )  # use data_T at time t to predict t
        elif self.flag_data_T == 0:
            M = calculate_conditional_expectation_dyncrep(
                self.gdata.adjacency_tensor[fold - 1], u, v, w, eta=eta, beta=beta
            )  # use data_T at time t-1 to predict t

        loglik_test = likelihood_conditional(
            M,
            beta,
            self.gdata.adjacency_tensor[fold],
            self.gdata.adjacency_tensor[fold - 1],
        )

        if fold > 1:
            M[self.gdata.adjacency_tensor[fold - 1].nonzero()] = (
                1 - beta
            )  # to calculate AUC

        comparison["auc"] = compute_link_prediction_AUC(
            self.gdata.adjacency_tensor[fold], M
        )
        comparison["loglik"] = loglik_test

        # Return the comparison dictionary
        return comparison

    def run_single_iteration(self):
        """
        Run the cross-validation procedure.
        """
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

        # Set up evaluation directory
        self.prepare_output_directory()

        # Prepare list to store results
        self.comparison = []

        # Import data
        self.load_data()

        # Make sure T is not too large
        self.T = max(0, min(self.T, self.gdata.adjacency_tensor.shape[0] - 1))

        logging.info("Starting the cross-validation procedure.")
        time_start = time.time()

        # Cross-validation loop
        for t in range(1, self.T + 1):  # skip first and last time step (last is hidden)
            if t == 1:
                self.parameters[
                    "fix_beta"
                ] = True  # for the first time step beta cannot be inferred
            else:
                self.parameters["fix_beta"] = False
            self.parameters["end_file"] = (
                self.end_file + "_" + str(t) + "_" + str(self.K)
            )

            # Prepare and run the algorithm
            tic = time.time()
            outputs, algorithm_object = self.prepare_and_run(t)

            # Output performance results
            self.comparison.append(
                self.calculate_performance_and_prepare_comparison(
                    outputs=outputs,
                    fold=t,
                    _mask=None,
                    algorithm_object=algorithm_object,
                )
            )

            logging.info("Time elapsed: %s seconds.", np.round(time.time() - tic, 2))

        logging.info(
            "\nTime elapsed: %s seconds.", np.round(time.time() - time_start, 2)
        )

        return self.comparison
