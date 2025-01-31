"""
This module contains the ACDCrossValidation class, which is used to perform cross-validation of the ACD algorithm.
"""

import logging
import pickle

import numpy as np

from probinet.evaluation.expectation_computation import (
    calculate_expectation_acd,
    calculate_Q_dense,
)

from ..evaluation.expectation_computation import compute_mean_lambda0
from ..evaluation.link_prediction import compute_link_prediction_AUC
from ..model_selection.cross_validation import CrossValidation
from ..model_selection.masking import extract_mask_kfold
from ..models.acd import AnomalyDetection


class ACDCrossValidation(CrossValidation):
    """
    Class for cross-validation of the ACD algorithm.
    """

    def __init__(
        self, algorithm, parameters, input_cv_params, numerical_parameters=None
    ):
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
            # Construct the evaluation file path for the mask
            outmask = self.out_folder + "mask_f" + str(fold) + "_" + self.adj + ".pkl"
            logging.debug("Mask saved in: %s", outmask)

            # Save the mask to a pickle file
            with open(outmask, "wb") as f:
                pickle.dump(np.where(mask > 0), f)

        # Return the mask
        return mask

    def prepare_and_run(self, mask):
        # Create a copy of the adjacency matrix B to use for training
        B_train = self.gdata.adjacency_tensor.copy()

        # Apply the mask to the training data by setting masked elements to 0
        B_train[mask > 0] = 0

        # Create a copy of gdata to use for training
        self.gdata_for_training = self.gdata._replace(adjacency_tensor=B_train)

        # Create an instance of the ACD algorithm
        algorithm_object = AnomalyDetection(**self.numerical_parameters)

        # Define rng from the seed and add it to the parameters
        self.parameters["rng"] = np.random.default_rng(seed=self.parameters["rseed"])

        # Fit the ACD models to the training data and get the outputs
        outputs = algorithm_object.fit(self.gdata_for_training, **self.parameters)

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
        M0 = compute_mean_lambda0(u, v, w)

        # Calculate the Q matrix if flag_anomaly is set
        if self.parameters["flag_anomaly"]:
            Q = calculate_Q_dense(self.gdata.adjacency_tensor, M0, pi, mu)
        else:
            Q = np.zeros_like(M0)

        # Calculate the expected matrix M using the parameters u, v, w, Q, and pi
        M = calculate_expectation_acd(u, v, w, Q, pi)

        # Calculate the AUC for the training set (where mask is not applied)
        comparison["aucA_train"] = compute_link_prediction_AUC(
            self.gdata.adjacency_tensor[0], M[0], mask=np.logical_not(mask[0])
        )

        # Calculate the AUC for the test set (where mask is applied)
        comparison["aucA_test"] = compute_link_prediction_AUC(
            self.gdata.adjacency_tensor[0], M[0], mask=mask[0]
        )

        # Return the comparison dictionary
        return comparison
