"""
This module contains the DynCRepCrossValidation class, which is used for cross-validation of the
DynCRep algorithm.
"""

import logging
import time

import numpy as np

from pgm.input.loader import import_data
from pgm.model.dyncrep import DynCRep
from pgm.model_selection.cross_validation import CrossValidation
from pgm.output.evaluate import calculate_AUC, calculate_conditional_expectation_dyncrep
from pgm.output.likelihood import likelihood_conditional


class DynCRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the DynCRep algorithm.

    - Hold-out the data at the latest time snapshot (at time T);
    - Infer parameters on the observed data (data up to time T-1);
    - Calculate performance measures in the hidden set (AUC).
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters=None):
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

    def extract_mask(self, fold):
        pass

    def load_data(self):
        """
        Load data from the input folder.
        """
        # Load data
        self.A, self.B, self.B_T, self.data_T_vals = import_data(
            self.in_folder + self.adj,
            sep=self.sep,
            header=0,
        )
        # Get the nodes
        self.nodes = self.A[0].nodes()

    def prepare_and_run(self, t):
        B_train = self.B[:t]  # use data up to time t-1 for training

        self.parameters["T"] = t
        # Initialize the ACD algorithm object
        algorithm_object = DynCRep(**self.num_parameters)

        outputs = algorithm_object.fit(
            data=B_train, nodes=self.nodes, **self.parameters
        )

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object # TODO: change signature to avoid unused 
            # variables?
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
            "rseed": algorithm_object.rseed,
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
                self.B[fold - 1], u, v, w, eta=eta, beta=beta
            )  # use data_T at time t-1 to predict t

        loglik_test = likelihood_conditional(M, beta, self.B[fold], self.B[fold - 1])

        if fold > 1:
            M[self.B[fold - 1].nonzero()] = 1 - beta  # to calculate AUC

        comparison["auc"] = calculate_AUC(M, self.B[fold])
        comparison["loglik"] = loglik_test

        # Return the comparison dictionary
        return comparison

    def run_single_iteration(self):
        """
        Run the cross-validation procedure.
        """
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

        # Set up output directory
        self.prepare_output_directory()

        # Prepare list to store results
        self.comparison = []

        # Import data
        self.load_data()

        # Make sure T is not too large
        self.T = max(0, min(self.T, self.B.shape[0] - 1))

        logging.info("Starting the cross-validation procedure.")
        time_start = time.time()

        # Cross-validation loop
        for t in range(1, self.T + 1):  # skip first and last time step (last is hidden)

            if t == 1:
                self.parameters["fix_beta"] = (
                    True  # for the first time step beta cannot be inferred
                )
            else:
                self.parameters["fix_beta"] = False
            self.parameters["end_file"] = (
                self.end_file + "_" + str(t) + "_" + str(self.K)
            )

            # Prepare and run the algorithm
            tic = time.time()
            outputs, algorithm_object = self.prepare_and_run(t)

            # Output performance results
            self.comparison.append(self.calculate_performance_and_prepare_comparison(
                outputs=outputs, mask=None, fold=t, algorithm_object=algorithm_object
            ))

            logging.info("Time elapsed: %s seconds.", np.round(time.time() - tic, 2))

        logging.info(
            "\nTime elapsed: %s seconds.", np.round(time.time() - time_start, 2)
        )

        return self.comparison
