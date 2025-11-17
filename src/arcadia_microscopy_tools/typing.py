import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt16Array = NDArray[np.uint16]
UByteArray = NDArray[np.ubyte]
ScalarArray = BoolArray | FloatArray | Int64Array | UInt16Array
