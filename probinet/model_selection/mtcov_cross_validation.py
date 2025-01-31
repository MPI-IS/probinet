"""
This module contains the MTCOVCrossValidation class, which is used for cross-validation of the MTCOV
algorithm.
"""

import logging
import pickle

import numpy as np

from ..evaluation.covariate_prediction import compute_covariate_prediction_accuracy
from ..evaluation.likelihood import loglikelihood
from ..evaluation.link_prediction import compute_multilayer_link_prediction_AUC
from ..models.classes import GraphData
from ..models.mtcov import MTCOV
from .cross_validation import CrossValidation
from .masking import extract_masks, shuffle_indicesG, shuffle_indicesX


class MTCOVCrossValidation(CrossValidation, MTCOV):
    """
    Class for cross-validation of the MTCOV algorithm.
    """

    def __init__(
        self, algorithm, parameters, input_cv_params, numerical_parameters=None
    ):
        """
        Constructor for the MTCOVCrossValidation class.
        Parameters
        ----------
        algorithm
        parameters
        input_cv_params
        """
        super().__init__(algorithm, parameters, input_cv_params, numerical_parameters)
        # These are the parameters for the MTCOV algorithm
        if numerical_parameters is None:
            numerical_parameters = {}
        self.parameters = parameters
        self.num_parameters = numerical_parameters

    def extract_mask(self, fold):
        # Prepare indices for cross-validation
        idxG = shuffle_indicesG(self.N, self.L, rng=self.rng)
        idxX = shuffle_indicesX(self.N, rng=self.rng)

        # Extract the masks for the current fold using k-fold cross-validation
        maskG, maskX = extract_masks(
            self.N,
            self.L,
            idxG=idxG,
            idxX=idxX,
            cv_type="kfold",
            NFold=self.NFold,
            fold=fold,
            rng=self.rng,
        )

        # If the out_mask attribute is set, save the masks to files
        if self.out_mask:
            outmaskG = f"{self.out_folder}maskG_f{fold}_{self.adj}.pkl"
            outmaskX = f"{self.out_folder}maskX_f{fold}_{self.adj}.pkl"
            logging.debug("Masks saved in: %s, %s", outmaskG, outmaskX)

            # Save the masks to pickle files
            with open(outmaskG, "wb") as f:
                pickle.dump(np.where(maskG > 0), f)
            with open(outmaskX, "wb") as f:
                pickle.dump(np.where(maskX > 0), f)

        # Return the masks
        return maskG, maskX

    def load_data(self):
        self.gdata: GraphData = MTCOV.load_data(
            self,
            self.in_folder,
            adj_name=self.adj,
            cov_name=self.cov,
            ego=self.ego,
            alter=self.alter,
            egoX=self.egoX,
            attr_name=self.attr_name,
            undirected=self.parameters["undirected"],
            force_dense=True,
            return_X_as_np=False,
        )
        # We need to create the attribute design matrix as a df
        self.Xs = self.gdata.design_matrix
        # But the algorithm expects it as a numpy array
        self.gdata = self.gdata._replace(design_matrix=np.array(self.Xs))

    def prepare_and_run(self, masks):
        maskG, maskX = masks
        # Create copies of the adjacency matrix B and covariate matrix X to use for training
        B_train = self.gdata.adjacency_tensor.copy()
        X_train = self.gdata.design_matrix.copy()

        # Apply the masks to the training data by setting masked elements to 0
        B_train[maskG > 0] = 0
        X_train[maskX > 0] = 0

        # Create a copy of gdata to use for training
        self.gdata_for_training = self.gdata._replace(adjacency_tensor=B_train)
        self.gdata_for_training = self.gdata_for_training._replace(
            design_matrix=X_train
        )

        # Initialize the MTCOV algorithm object
        algorithm_object = MTCOV(**self.num_parameters)

        # Fit the MTCOV models to the training data and get the outputs
        outputs = algorithm_object.fit(
            self.gdata_for_training,
            **{k: v for k, v in self.parameters.items() if k != "rseed"},
            rng=self.rng,
        )

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, masks, fold, _algorithm_object
    ):
        maskG, maskX = masks

        # Unpack the outputs from the algorithm
        U, V, W, BETA, logL = outputs

        # Initialize the comparison dictionary with keys as headers
        comparison = {
            "K": self.parameters["K"],
            "gamma": self.parameters["gamma"],
            "fold": fold,
            "rseed": self.rseed,
            "logL": logL,
        }

        # Calculate and assign the covariates accuracy values
        if self.parameters["gamma"] != 0:
            comparison["acc_train"] = compute_covariate_prediction_accuracy(
                self.Xs, U, V, BETA, mask=np.logical_not(maskX)
            )
            comparison["acc_test"] = compute_covariate_prediction_accuracy(
                self.Xs, U, V, BETA, mask=maskX
            )

        # Calculate and assign the AUC values
        if self.parameters["gamma"] != 1:
            comparison["auc_train"] = compute_multilayer_link_prediction_AUC(
                self.gdata.adjacency_tensor, U, V, W, mask=np.logical_not(maskG)
            )
            comparison["auc_test"] = compute_multilayer_link_prediction_AUC(
                self.gdata.adjacency_tensor, U, V, W, mask=maskG
            )

        # Calculate and assign the log-likelihood value
        comparison["logL_test"] = loglikelihood(
            self.gdata.adjacency_tensor,
            self.Xs,
            U,
            V,
            W,
            BETA,
            self.parameters["gamma"],
            maskG=maskG,
            maskX=maskX,
        )

        # Return the comparison dictionary
        return comparison
