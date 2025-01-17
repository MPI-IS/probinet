from pathlib import Path

RTOL = 1e-2
DECIMAL = 5
DECIMAL_2 = 4
DECIMAL_3 = 8
DECIMAL_4 = 20
DECIMAL_5 = 3
TOLERANCE_1 = 1e-3
TOLERANCE_2 = 1e-3
K_NEW = 5
RANDOM_SEED_REPROD = 0  # Random seed for reproducibility

CURRENT_FILE_PATH = Path(__file__)
INIT_STR = "_for_initialization"
PATH_FOR_INIT = CURRENT_FILE_PATH.parent / "inputs/"
PATH_TO_GT = CURRENT_FILE_PATH.parent / "inputs" / "cross_validation/"
