"""
This module contains the JointCRepCrossValidation class, which is used for cross-validation of the
JointCRep algorithm.
"""

from pgm.model.jointcrep import JointCRep
from pgm.model_selection.cross_validation import CrossValidation


class JointCRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the JointCRep algorithm.
    """

    def __init__(self, algorithm, parameters, input_cv_params, numerical_parameters={}):
        """
        Constructor for the JointCRepCrossValidation class.
        Parameters
        ----------
        algorithm
        parameters
        input_cv_params
        """
        super().__init__(algorithm, parameters, input_cv_params, numerical_parameters)
        # These are the parameters for the JointCRep algorithm
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

        # Initialize the JointCRep algorithm object
        algorithm_object = JointCRep(**self.num_parameters)

        # Fit the JointCRep model to the training data and get the outputs
        outputs = algorithm_object.fit(
            B_train, self.B_T, self.data_T_vals, nodes=self.nodes, **self.parameters
        )

        # Return the outputs and the algorithm object
        return outputs, algorithm_object

    def calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object
    ):
        super()._calculate_performance_and_prepare_comparison(
            outputs, mask, fold, algorithm_object
        )
