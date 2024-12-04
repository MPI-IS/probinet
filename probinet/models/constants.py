"""
This file contains the constants used in the models.
"""

from pathlib import Path

# Constants
EPS_ = 1e-12  # Small value to avoid division by zero
INF_ = 1e10  # Large value to represent infinity
ERR_MAX_ = 1e-12  # Maximum error allowed in the optimization algorithm
CONVERGENCE_TOL_ = 1e-4  # Convergence threshold for the optimization algorithm
ERR_ = 0.1  # Noise for the initialization
DECISION_ = 10  # Convergence parameter
OUTPUT_FOLDER = Path("outputs")
RTOL_DEFAULT = 1e-05
ATOL_DEFAULT = 1e-08
AG_DEFAULT = 1.5
BG_DEFAULT = 10.0
K_DEFAULT = 3
