from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .typing import ScalarArray


class ImageOperation:
    """A callable wrapper for image processing functions.

    Stores a method along with its args and kwargs for later execution on an image intensity array.
    Allows for convenient composition of image processing pipelines.
    """

    def __init__(self, method: callable, *args, **kwargs):
        """Create a new image operation.

        Args:
            method: The image processing function to wrap.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.
        """
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
    """

    operations: list[ImageOperation]

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the pipeline to an image.

        Args:
            intensities: Input image as an array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.
        """
        out = intensities.copy()
        for operation in self.operations:
            out = operation(out)
        return out


@dataclass
class PipelineParallelized:
    """A pipeline for parallel processing of multi-dimensional image data.

    Applies a sequence of image processing operations to each frame/slice in parallel
    using ThreadPoolExecutor. Parallelizes execution over the first dimension of the
    input array, with the last two dimensions assumed to be (y, x) spatial coordinates.

    Useful for timelapse data, z-stacks, multi-channel images, or any multi-dimensional
    image data where processing can be parallelized across the first axis.
    """

    operations: list[ImageOperation]
    max_workers: int = None

    def __call__(self, intensities: ScalarArray) -> ScalarArray:
        """Apply the pipeline to all frames/slices in parallel.

        Args:
            intensities: Input image as a multi-dimensional array of intensity values.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.
        """

        def process_frame(frame: ScalarArray) -> ScalarArray:
            """Apply all operations to a single frame."""
            out = frame.copy()
            for operation in self.operations:
                out = operation(out)
            return out

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            processed = list(executor.map(process_frame, intensities))

        return np.array(processed, dtype=float)
