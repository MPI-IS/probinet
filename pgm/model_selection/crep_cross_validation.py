import csv
import os

import numpy as np

from pgm.model.crep import CRep
from pgm.model_selection.cross_validation import CrossValidation
from pgm.output.evaluate import (
    calculate_AUC, calculate_conditional_expectation, calculate_expectation)
from pgm.output.likelihood import calculate_opt_func


class CRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the CRep algorithm.
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters={}):
        """
        Constructor for the CRepCrossValidation class.
        Parameters
        ----------
        algorithm
        parameters
        input_cv_params
        numerical_parameters
        """
        super().__init__(algorithm, parameters, input_cv_params, numerical_parameters)
        # These are the parameters for the CRep algorithm
        self.parameters = parameters
        self.num_parameters = numerical_parameters

    def extract_mask(self, fold):
        # Use the auxiliary method from the base class
        return super()._extract_mask(fold)

    def load_data(self):
        return super()._load_data()

    def prepare_and_run(self, mask):
        # Create a copy of the adjacency matrix B to use for training
        B_train = self.B.copy()

        # Apply the mask to the training data by setting masked elements to 0
        B_train[mask > 0] = 0

        # Initialize the CRep algorithm object
        algorithm_object = CRep(**self.num_parameters)

        # Fit the CRep model to the training data and get the outputs
        outputs = algorithm_object.fit(
            B_train, self.B_T, self.data_T_vals, nodes=self.nodes, **self.parameters
        )

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object
    ):
        # Unpack the outputs from the algorithm
        u, v, w, eta, maxL = outputs

        # Calculate the expected matrix M using the parameters u, v, w, and eta
        M = calculate_expectation(u, v, w, eta=eta)

        # Calculate the AUC for the training set (where mask is not applied)
        auc_train = calculate_AUC(M, self.B, mask=np.logical_not(mask))

        # Calculate the AUC for the test set (where mask is applied)
        auc_test = calculate_AUC(M, self.B, mask=mask)

        # Calculate the conditional expectation matrix M_cond
        M_cond = calculate_conditional_expectation(self.B, u, v, w, eta=eta)

        # Calculate the conditional AUC for the training set
        auc_cond_train = calculate_AUC(M_cond, self.B, mask=np.logical_not(mask))

        # Calculate the conditional AUC for the test set
        auc_cond_test = calculate_AUC(M_cond, self.B, mask=mask)

        # Calculate the optimization function value for the training set
        opt_func_train = calculate_opt_func(
            self.B,
            algorithm_object,
            mask=mask,
            assortative=self.parameters["assortative"],
        )

        # Initialize the comparison list with 11 elements
        comparison = [0 for _ in range(11)]

        # Assign the parameters to the first element of the comparison list
        comparison[0] = self.parameters["K"]

        # Assign the fold number and random seed to the second and third elements
        comparison[1], comparison[2] = fold, self.parameters["rseed"]

        # Assign eta to the fourth element
        comparison[3] = eta

        # Assign the AUC values to the fifth and sixth elements
        comparison[4] = auc_train
        comparison[5] = auc_test

        # Assign the conditional AUC values to the seventh and eighth elements
        comparison[6] = auc_cond_train
        comparison[7] = auc_cond_test

        # Assign the optimization function value to the tenth element
        comparison[9] = opt_func_train

        # Assign the maximum likelihood value to the ninth element
        comparison[8] = maxL

        # Assign the final iteration number of the algorithm to the eleventh element
        comparison[10] = algorithm_object.final_it

        # Store the comparison list in the instance variable
        self.comparison = comparison

    def save_results(self):
        # Check if the output file exists; if not, write the header
        if not os.path.isfile(self.out_file):  # write header
            with open(self.out_file, "w") as outfile:
                # Create a CSV writer object
                wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
                # Write the header row to the CSV file
                wrtr.writerow(
                    [
                        "K",
                        "fold",
                        "rseed",
                        "eta",
                        "auc_train",
                        "auc_test",
                        "auc_cond_train",
                        "auc_cond_test",
                        "opt_func_train",
                        "opt_func_test",
                        "max_it",
                    ]
                )
        # Open the output file in append mode
        outfile = open(self.out_file, "a")
        # Create a CSV writer object
        wrtr = csv.writer(outfile, delimiter=",", quotechar='"')
        # Write the comparison data to the CSV file
        wrtr.writerow(self.comparison)
        # Flush the output buffer to ensure all data is written to the file
        outfile.flush()
