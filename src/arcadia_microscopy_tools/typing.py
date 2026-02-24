import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
UByteArray = NDArray[np.uint8]
UInt16Array = NDArray[np.uint16]
Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]

ScalarArray = BoolArray | Float64Array | Int64Array | UInt16Array | UByteArray
