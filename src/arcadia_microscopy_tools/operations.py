from __future__ import annotations

import numpy as np
import skimage as ski

from .typing import FloatArray, ScalarArray


def rescale_intensity_by_percentiles(
    intensities: ScalarArray,
    percentile_range: tuple[float, float] = (0, 100),
    out_range: tuple[float, float] = (0, 1),
) -> FloatArray:
    """Rescale image intensities using percentile-based contrast stretching.

    Maps the intensity values from specified input percentile range to the output range.
    This is useful for normalizing images with varying intensity distributions.

    Args:
        intensities: Input image array.
        percentile_range:
            Tuple of (min, max) percentiles to use for intensity scaling.
            Default is (0, 100) which uses the full intensity range.
        out_range:
            Tuple of (min, max) values for the output intensity range.
            Default is (0, 1) for float normalization.

    Returns:
        FloatArray: Rescaled image with intensities mapped to the specified output range.

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
        in_range=(p1, p2),
        out_range=out_range,
    )


def get_background_subtracted_intensities(
    intensities: ScalarArray,
    percentile: float = 30.0,
    low_sigma: float = 0.6,
    high_sigma: float = 16.0,
) -> FloatArray:
    """Subtract background from image using difference of Gaussians and percentile thresholding.

    Applies difference of Gaussians filter to enhance features and then estimates and subtracts
    background based on a percentile threshold.

    Args:
        intensities: Input image array.
        percentile:
            Percentile of filtered image to use as background level (0-100).
            Lower values remove more background.
        low_sigma:
            Standard deviation for the smaller Gaussian kernel.
            Controls fine detail enhancement.
        high_sigma:
            Standard deviation for the larger Gaussian kernel.
            Controls background estimation extent.

    Returns:
        FloatArray: Background-subtracted image with negative values clipped to zero.

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
