import logging

from pgm.model_selection.acd_cross_validation import ACDCrossValidation
from pgm.model_selection.crep_cross_validation import CRepCrossValidation
from pgm.model_selection.dyncrep_cross_validation import DynCRepCrossValidation
from pgm.model_selection.jointcrep_cross_validation import JointCRepCrossValidation
from pgm.model_selection.mtcov_cross_validation import MTCOVCrossValidation
from pgm.model_selection.parameter_search import define_grid


def cross_validation(algorithm, model_parameters, cv_parameters, numerical_parameters={}):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    cv_classes = {
        "CRep": CRepCrossValidation,
        "JointCRep": JointCRepCrossValidation,
        "MTCOV": MTCOVCrossValidation,
        "ACD": ACDCrossValidation,
        "DynCRep": DynCRepCrossValidation,
    }

    if algorithm not in cv_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    logger.info("Starting cross-validation for algorithm: %s", algorithm)

    # Define the grid of parameters
    param_grid = define_grid(**model_parameters)
    logger.info("Parameter grid created with %d combinations", len(param_grid))

    # Loop over the grid of parameters
    for params in param_grid:
        logger.info("Running cross-validation for parameters: %s", params)
        # Instantiate the cross-validation class with the current parameters
        cv = cv_classes[algorithm](algorithm, params, cv_parameters, numerical_parameters)
        # Run the cross-validation
        cv.run_single_iteration()
        logger.info("Completed cross-validation for parameters: %s", params)

    logger.info("Completed cross-validation for algorithm: %s", algorithm)
