import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .typing import ScalarArray


class ImageOperation:
    """A callable wrapper for image processing functions.

    Stores a function along with its args and kwargs for later execution on an image
    intensity array. Allows for convenient composition of image processing pipelines.

    Args:
        func: The image processing function to wrap.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """

    __slots__ = ("func", "args", "kwargs")

    def __init__(self, func: Callable[..., ScalarArray], *args: object, **kwargs: object) -> None:
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ImageOperation instances are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ImageOperation instances are immutable")

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the operation to an image.

        Args:
            intensities: Input image as an array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array.
        """
        return self.func(intensities, *self.args, **self.kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageOperation):
            return NotImplemented
        return self.func == other.func and self.args == other.args and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return hash((self.func, self.args, tuple(sorted(self.kwargs.items()))))

    def __repr__(self) -> str:
        """Create a string representation of the operation."""
        args_repr = [repr(arg) for arg in self.args]
        kwargs_repr = [f"{key}={repr(value)}" for key, value in self.kwargs.items()]
        args_kwargs_repr = ", ".join(args_repr + kwargs_repr)
        return f"{self.func.__name__}({args_kwargs_repr})"


@dataclass
class Pipeline:
    """A sequence of image processing operations.

    Combines multiple image operations into a single callable pipeline that applies each
    operation in sequence to an input image. Supports optional parallelization over the
    first axis of multi-dimensional data (e.g., timelapse, z-stacks).

    Attributes:
        operations: List of ImageOperation instances to apply in sequence.
        copy: If True, creates a copy of the input array before processing. If False,
            operations are applied directly to the input. Default is False for performance.
            Ignored when parallel=True (the output is always a new array).
        preserve_dtype: If True, forces output to have the same dtype as input. If False,
            allows dtype to change based on operations (e.g., uint16 -> float64 for
            normalization). Default is False.
        parallel: If True, applies operations to each slice along the first axis in
            parallel using ThreadPoolExecutor. Useful for timelapse, z-stack, or
            multi-channel data. Requires at least 3D input. Default is False.
        max_workers: Maximum number of worker threads when parallel=True. Must be at
            least 1. If None, ThreadPoolExecutor uses its default (typically the number
            of CPU cores). Ignored when parallel=False.

    Note:
        Parallel mode uses thread-based parallelism, which is most effective for operations
        that release the GIL (like numpy operations). Pure Python operations may not benefit
        from parallelization due to the Global Interpreter Lock.
    """

    operations: list[ImageOperation]
    copy: bool = False
    preserve_dtype: bool = False
    parallel: bool = False
    max_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate the pipeline configuration."""
        if isinstance(self.operations, tuple):
            self.operations = list(self.operations)
        if not self.operations:
            raise ValueError("Pipeline must have at least one operation")
        if not all(callable(op) for op in self.operations):
            raise TypeError("All operations must be callable (wrap functions with ImageOperation)")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        if self.parallel and self.copy:
            warnings.warn(
                "copy=True has no effect when parallel=True. "
                "Parallel mode always produces a new output array.",
                UserWarning,
                stacklevel=2,
            )

    def _apply_operations(self, intensities: ScalarArray) -> ScalarArray:
        """Apply all operations to an image array."""
        out = intensities.copy() if self.copy else intensities
        for operation in self.operations:
            out = operation(out)
        return out

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the pipeline to an image.

        When parallel=False, applies operations to the entire array sequentially.
        When parallel=True, maps operations over each slice of the first axis using
        a thread pool. Requires at least 3D input.

        Args:
            intensities: Input image as an array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.

        Raises:
            ValueError: If parallel=True and input has fewer than 3 dimensions.
        """
        if self.parallel:
            if intensities.ndim < 3:
                raise ValueError(
                    f"Parallel mode requires at least 3D input (got {intensities.ndim}D). "
                    "The first axis is used to distribute work across threads."
                )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed = list(executor.map(self._apply_operations, intensities))
            if self.preserve_dtype:
                return np.array(processed, dtype=intensities.dtype)  # type: ignore
            return np.array(processed)  # type: ignore

        result = self._apply_operations(intensities)
        if self.preserve_dtype and result.dtype != intensities.dtype:
            return result.astype(intensities.dtype)  # type: ignore
        return result

    def __len__(self) -> int:
        """Return the number of operations in the pipeline."""
        return len(self.operations)

    def __repr__(self) -> str:
        """Create a string representation of the pipeline."""
        operations_repr = ", ".join(repr(operation) for operation in self.operations)
        params = []
        if self.copy:
            params.append("copy=True")
        if self.preserve_dtype:
            params.append("preserve_dtype=True")
        if self.parallel:
            params.append("parallel=True")
        if self.max_workers is not None:
            params.append(f"max_workers={self.max_workers}")
        params_str = f", {', '.join(params)}" if params else ""
        return f"Pipeline([{operations_repr}]{params_str})"
