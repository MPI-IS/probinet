"""
This module contains the ACDCrossValidation class, which is used to perform cross-validation of the ACD algorithm.
"""

import logging
import pickle

import numpy as np

from pgm.input.loader import import_data
from pgm.model.acd import AnomalyDetection
from pgm.model_selection.cross_validation import CrossValidation
from pgm.model_selection.masking import extract_mask_kfold
from pgm.output.evaluate import (
    calculate_AUC, calculate_expectation_acd, calculate_Q_dense, lambda0_full)


class ACDCrossValidation(CrossValidation):
    """
    Class for cross-validation of the ACD algorithm.
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters={}):
        """
        Constructor for the ACDCrossValidation class.
        Parameters
        ----------
        algorithm
        parameters
        input_cv_params
        numerical_parameters
        """
        super().__init__(algorithm, parameters, input_cv_params, numerical_parameters)
        # These are the parameters for the ACD algorithm
        self.parameters = parameters
        self.numerical_parameters = numerical_parameters

    def extract_mask(self, fold):
        # Extract the mask for the current fold using k-fold cross-validation
        mask = extract_mask_kfold(self.indices, self.N, fold=fold, NFold=self.NFold)

        # If the out_mask attribute is set, save the mask to a file
        if self.out_mask:
            # Construct the output file path for the mask
            outmask = self.out_folder + "mask_f" + str(fold) + "_" + self.adj + ".pkl"
            logging.debug("Mask saved in: %s", outmask)

            # Save the mask to a pickle file
            with open(outmask, "wb") as f:
                pickle.dump(np.where(mask > 0), f)

        # Return the mask
        return mask

    def load_data(self):
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

    def prepare_and_run(self, mask):
        # Create a copy of the adjacency matrix B to use for training
        B_train = self.B.copy()

        # Apply the mask to the training data by setting masked elements to 0
        B_train[mask > 0] = 0

        # Create an instance of the ACD algorithm
        algorithm_object = AnomalyDetection(**self.numerical_parameters)

        # Fit the ACD model to the training data and get the outputs
        outputs = algorithm_object.fit(B_train, nodes=self.nodes, **self.parameters)

        # Return the outputs and the algorithm object

        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object
    ):
        # Unpack the outputs from the algorithm
        u, v, w, mu, pi, maxELBO = outputs

        # Initialize the comparison dictionary with keys as headers
        comparison = {
            "K": self.parameters["K"],
            "fold": fold,
            "rseed": self.rseed,
            "flag_anomaly": self.parameters["flag_anomaly"],
            "mu": mu,
            "pi": pi,
            "ELBO": maxELBO,
            "final_it": algorithm_object.final_it,
        }

        # Calculate the expected matrix M0 using the parameters u, v, and w
        M0 = lambda0_full(u, v, w)

        # Calculate the Q matrix if flag_anomaly is set
        if self.parameters["flag_anomaly"]:
            Q = calculate_Q_dense(self.B, M0, pi, mu)
        else:
            Q = np.zeros_like(M0)

        # Calculate the expected matrix M using the parameters u, v, w, Q, and pi
        M = calculate_expectation_acd(u, v, w, Q, pi)

        # Calculate the AUC for the training set (where mask is not applied)
        comparison["aucA_train"] = calculate_AUC(
            M[0], self.B[0], mask=np.logical_not(mask[0])
        )

        # Calculate the AUC for the test set (where mask is applied)
        comparison["aucA_test"] = calculate_AUC(M[0], self.B[0], mask=mask[0])

        # Store the comparison dictionary values in the instance variable as a list
        self.comparison = comparison
