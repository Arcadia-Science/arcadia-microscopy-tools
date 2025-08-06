from dataclasses import dataclass

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
            intensities: The input image intensity array.

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
            intensities: The input image intensity array.

        Returns:
            ScalarArray: The processed image intensity array after applying all operations.
        """
        out = intensities.copy()
        for operation in self.operations:
            out = operation(out)
        return out
