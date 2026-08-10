from typing import Any

import numpy as np
from numpy.typing import NDArray

# Type aliases for numpy arrays with a specific, guaranteed scalar dtype.
BoolArray = NDArray[np.bool_]
UByteArray = NDArray[np.uint8]
UInt16Array = NDArray[np.uint16]
Int64Array = NDArray[np.int64]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

# General-purpose image array whose scalar dtype is not statically constrained.
ScalarArray = NDArray[Any]
