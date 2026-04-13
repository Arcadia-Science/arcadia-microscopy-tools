import numpy as np
from numpy.typing import NDArray

# Type aliases for common numpy array types
BoolArray = NDArray[np.bool_]
UByteArray = NDArray[np.uint8]
UInt16Array = NDArray[np.uint16]
Int64Array = NDArray[np.int64]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

# Union type for arrays with numeric or boolean scalar types
ScalarArray = BoolArray | UByteArray | UInt16Array | Int64Array | Float32Array | Float64Array
