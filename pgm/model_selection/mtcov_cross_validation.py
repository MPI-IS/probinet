"""
This module contains the MTCOVCrossValidation class, which is used for cross-validation of the MTCOV
algorithm.
"""

import logging
import pickle

import numpy as np

from pgm.input.loader import import_data_mtcov
from pgm.model.mtcov import MTCOV
from pgm.model_selection.cross_validation import CrossValidation
from pgm.model_selection.masking import extract_masks, shuffle_indicesG, shuffle_indicesX
from pgm.model_selection.metrics import covariates_accuracy
from pgm.output.evaluate import calculate_AUC_mtcov
from pgm.output.likelihood import loglikelihood


class MTCOVCrossValidation(CrossValidation):
    """
    Class for cross-validation of the MTCOV algorithm.
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters=None):
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
        idxG = shuffle_indicesG(self.N, self.L, rseed=self.rseed)
        idxX = shuffle_indicesX(self.N, rseed=self.rseed)

        # Extract the masks for the current fold using k-fold cross-validation
        maskG, maskX = extract_masks(
            self.N,
            self.L,
            idxG=idxG,
            idxX=idxX,
            cv_type="kfold",
            NFold=self.NFold,
            fold=fold,
            rseed=self.rseed,
            out_mask=self.out_mask,
        )

        # If the out_mask attribute is set, save the masks to files
        if self.out_mask:
            outmaskG = self.out_folder + "maskG_f" + str(fold) + "_" + self.adj + ".pkl"
            outmaskX = self.out_folder + "maskX_f" + str(fold) + "_" + self.adj + ".pkl"
            logging.debug("Masks saved in: %s, %s", outmaskG, outmaskX)

            # Save the masks to pickle files
            with open(outmaskG, "wb") as f:
                pickle.dump(np.where(maskG > 0), f)
            with open(outmaskX, "wb") as f:
                pickle.dump(np.where(maskX > 0), f)

        # Return the masks
        return maskG, maskX

    def load_data(self):
        # Load data
        self.A, self.B, self.X, self.nodes = import_data_mtcov(
            self.in_folder,
            adj_name=self.adj,
            cov_name=self.cov,
            ego=self.ego,
            alter=self.alter,
            egoX=self.egoX,
            attr_name=self.attr_name,
            undirected=self.parameters["undirected"],
            force_dense=True,
        )
        # Convert X to a numpy array
        self.Xs = np.array(self.X)

    def prepare_and_run(self, masks):
        maskG, maskX = masks
        # Create copies of the adjacency matrix B and covariate matrix X to use for training
        B_train = self.B.copy()
        X_train = self.Xs.copy()

        # Apply the masks to the training data by setting masked elements to 0
        B_train[maskG > 0] = 0
        X_train[maskX > 0] = 0

        # Initialize the MTCOV algorithm object
        algorithm_object = MTCOV(**self.num_parameters)

        # Fit the MTCOV model to the training data and get the outputs
        outputs = algorithm_object.fit(
            B_train,
            X_train,
            nodes=self.nodes,
            **{k: v for k, v in self.parameters.items() if k != "rseed"},
            rseed=self.rseed,
        )

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, masks, fold, algorithm_object
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
            comparison["acc_train"] = covariates_accuracy(
                self.X, U, V, BETA, mask=np.logical_not(maskX)
            )
            comparison["acc_test"] = covariates_accuracy(self.X, U, V, BETA, mask=maskX)

        # Calculate and assign the AUC values
        if self.parameters["gamma"] != 1:
            comparison["auc_train"] = calculate_AUC_mtcov(
                self.B, U, V, W, mask=np.logical_not(maskG)
            )
            comparison["auc_test"] = calculate_AUC_mtcov(self.B, U, V, W, mask=maskG)

        # Calculate and assign the log-likelihood value
        comparison["logL_test"] = loglikelihood(
            self.B,
            self.X,
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
