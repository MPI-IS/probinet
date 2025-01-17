"""
Custom type hints.
'"""

from os import PathLike
from typing import Sequence, Union

import numpy as np
from sparse import COO

EndFileType = str
FilesType = PathLike
GraphDataType = Union[COO, np.ndarray]
MaskType = np.ndarray
SubsNzType = tuple[int, int, int]
ArraySequence = Sequence[np.ndarray]
