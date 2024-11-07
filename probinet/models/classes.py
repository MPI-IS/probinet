"""
This module contains the definition of namedtuple classes.
"""

from collections import namedtuple

GraphData = namedtuple(
    "GraphData",
    [
        "graph_list",
        "adjacency_tensor",
        "transposed_tensor",
        "data_values",
        "nodes",
        "design_matrix",
    ],
    defaults=[None, None, None, None, None, None],
)
