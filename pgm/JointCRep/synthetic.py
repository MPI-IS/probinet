""" Code to generate synthetic networks that emulates directed networks (possibly weighted)
with or without reciprocity. Self-loops are removed and only the largest connected component is considered. """

import os
import math
import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from abc import ABCMeta



# Repeated? It was in tools.py too.
# def transpose_tensor(M):
#     """
#         Compute the transpose of a tensor with respect to the second and third dimensions.
#
#         INPUT
#         ----------
#         M : ndarray
#             Numpy tensor.
#
#         OUTPUT
#         -------
#         Transpose of the matrix.
#     """
#
#     return np.einsum("aij->aji", M)
