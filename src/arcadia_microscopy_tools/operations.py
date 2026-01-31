from __future__ import annotations
from typing import Literal

import numpy as np
import skimage as ski

from .typing import BoolArray, Float64Array, ScalarArray


def rescale_by_percentile(
    intensities: ScalarArray,
    percentile_range: tuple[float, float] = (0, 100),
    out_range: tuple[float, float] = (0, 1),
) -> ScalarArray:
    """Rescale image intensities using percentile-based contrast stretching.

    Maps the intensity values from specified input percentile range to the output range.
    This is useful for normalizing images with varying intensity distributions.

    Args:
        intensities: Input image array.
        percentile_range: Tuple of (min, max) percentiles to use for intensity scaling.
            Default is (0, 100) which uses the full intensity range.
        out_range: Tuple of (min, max) values for the output intensity range.
            Default is (0, 1) for float normalization.

    Returns:
        ScalarArray: Rescaled image with intensities mapped to the specified output range.

    Raises:
        ValueError: If percentile_range or out_range values are invalid.
    """
    # Validate input parameters
    if not (0 <= percentile_range[0] < percentile_range[1] <= 100):
        raise ValueError(
            f"Invalid percentile range: {percentile_range}. "
            f"Values must be in ascending order between 0 and 100."
        )

    # Handle empty or constant images
    if intensities.size == 0:
        return np.zeros_like(intensities, dtype=float)
    if np.min(intensities) == np.max(intensities):
        return np.full_like(intensities, out_range[0], dtype=float)

    # Calculate percentiles
    p1, p2 = np.percentile(intensities, percentile_range)

    # Apply rescaling
    return ski.exposure.rescale_intensity(
        intensities,
        in_range=(p1, p2),  # type: ignore
        out_range=out_range,  # type: ignore
    )


def subtract_background_dog(
    intensities: ScalarArray,
    low_sigma: float = 0.6,
    high_sigma: float = 16.0,
    percentile: float = 0,
) -> Float64Array:
    """Subtract background from image using difference of Gaussians and percentile thresholding.

    Applies difference of Gaussians filter to enhance features and then estimates and subtracts
    background based on a percentile threshold.

    Args:
        intensities: Input image array.
        low_sigma: Standard deviation for the smaller Gaussian kernel.
            Controls fine detail enhancement. Default is 0.6.
        high_sigma: Standard deviation for the larger Gaussian kernel.
        Controls background estimation extent. Default is 16.
        percentile: Percentile of filtered image to use as background level (0-100).
            Default is 0 (minimum value).

    Returns:
        Float64Array: Background-subtracted image with negative values clipped to zero.

    Notes:
        - For best results, low_sigma should be smaller than the smallest feature of interest,
          and high_sigma should be larger than the largest feature.
    """
    # Validate inputs
    if not (0 <= percentile <= 100):
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    if low_sigma >= high_sigma:
        raise ValueError(f"low_sigma ({low_sigma}) must be smaller than high_sigma ({high_sigma})")

    # Apply difference of Gaussians filter
    intensities_dog = ski.filters.difference_of_gaussians(intensities, low_sigma, high_sigma)

    # Estimate background level
    background_level = np.percentile(intensities_dog, percentile)

    # Subtract background and clip negative values
    return np.clip(intensities_dog - background_level, 0, None)


def crop_to_center(
    intensities: ScalarArray,
    output_shape: tuple[int, int],
) -> ScalarArray:
    """Crop image to specified dimensions by taking pixels from the center.

    Extracts a rectangular region from the center of the input image with the
    specified output dimensions. If the requested crop size exceeds the image
    dimensions, the crop is limited to the available image size.

    Args:
        intensities: Input image array.
        output_shape: Tuple of (height, width) for the desired output dimensions.

    Returns:
        ScalarArray: Cropped image centered on the original image.

    Notes:
        - Works with multi-dimensional arrays where the last two dimensions are height and width.
        - If output_shape is larger than the image, returns the full image.
        - Cropping is always centered, with equal padding removed from opposite sides.
    """
    height, width = intensities.shape[-2:]
    crop_height, crop_width = output_shape

    # Ensure the crop size does not exceed the image size
    crop_width = min(width, crop_width)
    crop_height = min(height, crop_height)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    return intensities[..., top : top + crop_height, left : left + crop_width]


def apply_threshold(
    intensities: ScalarArray,
    method: Literal[
        "otsu",
        "li",
        "yen",
        "isodata",
        "mean",
        "minimum",
        "triangle",
        "local",
        "niblack",
        "sauvola",
    ] = "otsu",
    **kwargs,
) -> BoolArray:
    """Apply thresholding to convert grayscale image to binary using various methods.

    Uses threshold calculation methods from skimage.filters to determine an optimal
    threshold value, then applies it to create a binary image.

    Args:
        intensities: Input grayscale image array.
        method: Thresholding method to use. Supported methods include:
            - 'otsu': Otsu's method (default)
            - 'li': Li's minimum cross entropy method
            - 'yen': Yen's method
            - 'isodata': ISODATA method
            - 'mean': Mean-based threshold
            - 'minimum': Minimum method
            - 'triangle': Triangle algorithm
            - 'local': Adaptive local threshold
            - 'niblack': Niblack local threshold
            - 'sauvola': Sauvola local threshold
        **kwargs: Additional keyword arguments passed to the thresholding function.
            For local methods (niblack, sauvola, local), common kwargs include:
            - window_size: Size of the local neighborhood
            - k: Parameter controlling threshold adjustment

    Returns:
        BoolArray: Binary image where pixels above threshold are True.

    Raises:
        ValueError: If the specified method is not supported.

    Examples:
        >>> binary = apply_threshold(image, method='otsu')
        >>> binary = apply_threshold(image, method='sauvola', window_size=25)
    """
    # Map method names to skimage.filters threshold functions
    threshold_methods = {
        "otsu": ski.filters.threshold_otsu,
        "li": ski.filters.threshold_li,
        "yen": ski.filters.threshold_yen,
        "isodata": ski.filters.threshold_isodata,
        "mean": ski.filters.threshold_mean,
        "minimum": ski.filters.threshold_minimum,
        "triangle": ski.filters.threshold_triangle,
        "local": ski.filters.threshold_local,
        "niblack": ski.filters.threshold_niblack,
        "sauvola": ski.filters.threshold_sauvola,
    }

    # Handle empty or constant images
    if intensities.size == 0:
        return np.zeros_like(intensities, dtype=bool)
    if np.min(intensities) == np.max(intensities):
        return np.zeros_like(intensities, dtype=bool)

    method_lower = method.lower()
    if method_lower not in threshold_methods:
        raise ValueError(
            f"Unsupported thresholding method: '{method}'. "
            f"Supported methods: {', '.join(threshold_methods.keys())}"
        )

    threshold_func = threshold_methods[method_lower]

    # Local methods (niblack, sauvola) return threshold array, others return scalar
    threshold_value = threshold_func(intensities, **kwargs)

    return intensities > threshold_value
