"""
This module contains the CRepCrossValidation class, which is used for cross-validation of the CRep algorithm.
"""

from pgm.model.crep import CRep
from pgm.model_selection.cross_validation import CrossValidation


class CRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the CRep algorithm.
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters=None):
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
        if numerical_parameters is None:
            numerical_parameters = {}
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
        return super()._calculate_performance_and_prepare_comparison(
            outputs, mask, fold, algorithm_object
        )
