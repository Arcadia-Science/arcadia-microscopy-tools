from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .typing import ScalarArray


class ImageOperation:
    """A callable wrapper for image processing functions.

    Stores a method along with its args and kwargs for later execution on an image intensity array.
    Allows for convenient composition of image processing pipelines.

    Args:
        method: The image processing function to wrap.
        *args: Positional arguments to pass to the method.
        **kwargs: Keyword arguments to pass to the method.
    """

    def __init__(self, method: Callable, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the operation to an image.

        Args:
            intensities: Input image as an array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array.
        """
        return self.method(intensities, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        """Create a string representation of the operation."""
        args_repr = [repr(arg) for arg in self.args]
        kwargs_repr = [f"{key}={repr(value)}" for key, value in self.kwargs.items()]
        args_kwargs_repr = ", ".join(args_repr + kwargs_repr)
        return f"{self.method.__name__}({args_kwargs_repr})"


@dataclass
class Pipeline:
    """A sequence of image processing operations.

    Combines multiple image operations into a single callable pipeline that applies each operation
    in sequence to an input image.

    Attributes:
        operations: List of ImageOperation instances to apply in sequence.
        copy: If True, creates a copy of the input array before processing. If False,
            operations are applied directly to the input. Default is False for performance.
    """

    operations: list[ImageOperation]
    copy: bool = False

    def __post_init__(self):
        """Validate the pipeline configuration."""
        if not self.operations:
            raise ValueError("Pipeline must have at least one operation")

    def _apply_operations(self, intensities: ScalarArray) -> ScalarArray:
        """Apply all operations to an image array."""
        out = intensities.copy() if self.copy else intensities
        for operation in self.operations:
            out = operation(out)
        return out

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the pipeline to an image.

        Args:
            intensities: Input image as an array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.
        """
        return self._apply_operations(intensities)

    def __len__(self) -> int:
        """Return the number of operations in the pipeline."""
        return len(self.operations)

    def __repr__(self) -> str:
        """Create a string representation of the pipeline."""
        operations_repr = ", ".join(repr(operation) for operation in self.operations)
        copy_str = ", copy=True" if self.copy else ""
        return f"Pipeline([{operations_repr}]{copy_str})"


@dataclass
class PipelineParallelized:
    """A pipeline for parallel processing of multi-dimensional image data.

    Applies a sequence of image processing operations to each frame/slice in parallel
    using ThreadPoolExecutor. Parallelizes execution over the first dimension of the
    input array, with the last two dimensions assumed to be (y, x) spatial coordinates.

    Useful for timelapse data, z-stacks, multi-channel images, or any multi-dimensional
    image data where processing can be parallelized across the first axis.

    Attributes:
        operations: List of ImageOperation instances to apply in sequence.
        max_workers: Maximum number of worker threads for parallel processing. If None,
            ThreadPoolExecutor will use its default (typically number of CPU cores).
        copy: If True, creates a copy of each frame before processing. If False,
            operations are applied directly to each frame. Default is False for performance.

    Note:
        Uses thread-based parallelism, which is most effective for operations that release
        the GIL (like numpy operations). Pure Python operations may not benefit from
        parallelization due to the Global Interpreter Lock.
    """

    operations: list[ImageOperation]
    max_workers: int | None = None
    copy: bool = False

    def __post_init__(self):
        """Validate the pipeline configuration."""
        if not self.operations:
            raise ValueError("Pipeline must have at least one operation")

    def _apply_operations(self, intensities: ScalarArray) -> ScalarArray:
        """Apply all operations to an image array."""
        out = intensities.copy() if self.copy else intensities
        for operation in self.operations:
            out = operation(out)
        return out

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the pipeline to all frames/slices in parallel.

        Args:
            intensities: Input image as a multi-dimensional array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            processed = list(executor.map(self._apply_operations, intensities))

        return np.array(processed, dtype=intensities.dtype)

    def __len__(self) -> int:
        """Return the number of operations in the pipeline."""
        return len(self.operations)

    def __repr__(self) -> str:
        """Create a string representation of the pipeline."""
        operations_repr = ", ".join(repr(operation) for operation in self.operations)
        params = []
        if self.max_workers is not None:
            params.append(f"max_workers={self.max_workers}")
        if self.copy:
            params.append("copy=True")
        params_str = f", {', '.join(params)}" if params else ""
        return f"PipelineParallelized([{operations_repr}]{params_str})"
