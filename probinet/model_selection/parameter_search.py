"""
This module defines the grid of parameters to be tested for models selection.
It provides a function to generate all possible combinations of parameter values.
"""

from itertools import product
from typing import Any, Dict, List


def define_grid(**kwargs: Any) -> List[Dict[str, Any]]:
    """
    Define the grid of parameters to be tested.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments where the key is the parameter name and the value is a list of possible values for that parameter.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each representing a unique combination of parameter values.
    """
    # Check that each value is a list
    if not all(isinstance(v, list) for v in kwargs.values()):
        raise ValueError(
            "Some values are not lists:\n\t"
            + "\n\t".join(
                f"{k} : {v}" for k, v in kwargs.items() if not isinstance(v, list)
            )
        )

    # Extract parameter names
    param_names = kwargs.keys()
    # Extract parameter values
    param_values = kwargs.values()
    # Generate all combinations of parameter values
    param_combinations = product(*param_values)
    # Create a list of dictionaries for each combination
    grid = [dict(zip(param_names, combination)) for combination in param_combinations]
    return grid
