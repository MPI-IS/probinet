"""
Main module for running cross-validation for different algorithms.
"""

import logging
from typing import Any, Optional

import pandas as pd

from .acd_cross_validation import ACDCrossValidation
from .crep_cross_validation import CRepCrossValidation
from .dyncrep_cross_validation import DynCRepCrossValidation
from .jointcrep_cross_validation import JointCRepCrossValidation
from .mtcov_cross_validation import MTCOVCrossValidation
from .parameter_search import define_grid


def cross_validation(
    algorithm: str,
    model_parameters: dict[str, Any],
    cv_parameters: dict[str, Any],
    numerical_parameters: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run cross-validation for a given algorithm.
    Parameters
    ----------
    algorithm
        String with the name of the algorithm to run.
    model_parameters
        Dictionary with the parameters for the algorithm.
    cv_parameters
        Dictionary with the parameters for the cross-validation.
    numerical_parameters
        Dictionary with the numerical parameters for the algorithm, like the number of iterations, etc.

    Returns
    -------
    results_df
        DataFrame with the results of the cross-validation.
    """
    if numerical_parameters is None:
        numerical_parameters = {}
    cv_classes = {
        "CRep": CRepCrossValidation,
        "JointCRep": JointCRepCrossValidation,
        "MTCOV": MTCOVCrossValidation,
        "ACD": ACDCrossValidation,
        "DynCRep": DynCRepCrossValidation,
    }

    if algorithm not in cv_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    logging.info("Starting cross-validation for algorithm: %s", algorithm)

    # If one of the parameters is an element, convert it to a list
    for key, value in model_parameters.items():
        if not isinstance(value, list):
            logging.debug(
                "Converting parameter %s to list. The value used is %s", key, value
            )
            model_parameters[key] = [value]

    # Define the grid of parameters
    param_grid = define_grid(**model_parameters)
    logging.info("Parameter grid created with %d combinations", len(param_grid))

    # Define list to store results
    results = []

    # Loop over the grid of parameters
    for params in param_grid:
        logging.info("Running cross-validation for parameters: %s", params)
        # Instantiate the cross-validation class with the current parameters
        cv = cv_classes[algorithm](
            algorithm, params, cv_parameters, numerical_parameters
        )
        # Run the cross-validation and store the results
        # The evaluation of run_single_iteration is a list of dictionaries with the results for the
        # different folds. Here, we add those lists to a results list.
        results += cv.run_single_iteration()
        logging.info("Completed cross-validation for parameters: %s", params)

    # Transform the list of results into a DataFrame
    results_df = pd.DataFrame(results)

    logging.info("Completed cross-validation for algorithm: %s", algorithm)

    # If the evaluation file is set, save the results to a CSV file
    if cv.out_folder:
        # Prepare the evaluation directory
        adjacency = cv.prepare_file_name()
        out_file = cv.out_folder + adjacency + "_cv.csv"
        # Save the results to a CSV file
        results_df.to_csv(out_file, index=False)
        logging.info("Results saved in: %s", out_file)

    return results_df
