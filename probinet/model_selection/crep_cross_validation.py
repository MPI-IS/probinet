"""
This module contains the CRepCrossValidation class, which is used for cross-validation of the CRep algorithm.
"""

from ..models.crep import CRep
from .cross_validation import CrossValidation


class CRepCrossValidation(CrossValidation):
    """
    Class for cross-validation of the CRep algorithm.
    """

    def __init__(
        self, algorithm, parameters, input_cv_params, numerical_parameters=None
    ):
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
        self.model = CRep

    def extract_mask(self, fold):
        # Use the auxiliary method from the base class
        return super()._extract_mask(fold)

    def calculate_performance_and_prepare_comparison(
        self, outputs, mask, fold, algorithm_object
    ):
        return super()._calculate_performance_and_prepare_comparison(
            outputs, mask, fold, algorithm_object
        )
