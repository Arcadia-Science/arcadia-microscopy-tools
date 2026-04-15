from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage.color import gray2rgb

from .channels import Channel
from .typing import Float64Array


@dataclass
class Layer:
    """A single layer in a fluorescence overlay.

    Args:
        channel: Channel providing color and identity.
        intensities: 2D array of intensity values in [0, 1].
        opacity: Global opacity multiplier in [0, 1]. Default is 1 (fully opaque).
        transparent: If True (default), colormap fades from transparent to channel color.
            If False, colormap fades from black to channel color.
    """

    channel: Channel
    intensities: Float64Array
    opacity: float = 1.0
    transparent: bool = True

    def __post_init__(self) -> None:
        if self.intensities.ndim != 2:
            raise ValueError(f"Expected 2D intensities array, got shape {self.intensities.shape}")
        if not 0 <= self.opacity <= 1:
            raise ValueError(f"Opacity must be in [0, 1], got {self.opacity}")


def overlay_channels(
    background: Float64Array,
    channel_intensities: dict[Channel, Float64Array],
    opacity: float = 1.0,
    transparent: bool = True,
) -> Float64Array:
    """Create a fluorescence overlay with uniform settings for all channels.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        channel_intensities: Dict mapping Channel objects to their 2D intensity arrays
            (values in [0, 1]).
        opacity: Global opacity multiplier for all channels. Default is 1 (fully opaque).
        transparent: If True (default), all colormaps fade from transparent to channel color.
            If False, all colormaps fade from black to channel color.

    Returns:
        RGB image (HxWx3 array) with all channels alpha-blended onto background.

    Example:
        >>> overlay = overlay_channels(
        ...     background=brightfield,
        ...     channel_intensities={
        ...         DAPI: dapi_intensities,
        ...         FITC: fitc_intensities,
        ...         TRITC: tritc_intensities,
        ...     }
        ... )
    """
    layers = [
        Layer(channel, intensities, opacity, transparent)
        for channel, intensities in channel_intensities.items()
    ]
    return create_sequential_overlay(background, layers)


def create_sequential_overlay(
    background: Float64Array,
    layers: list[Layer],
) -> Float64Array:
    """Create an overlay by sequentially blending layers onto a background.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        layers: List of Layer objects to overlay in sequence.

    Returns:
        RGB image (HxWx3 array) with all layers alpha-blended onto background.

    Raises:
        ValueError: If background is not a 2D array.

    Example:
        >>> overlay = create_sequential_overlay(
        ...     background=brightfield,
        ...     layers=[
        ...         Layer(DAPI, dapi_intensities),
        ...         Layer(FITC, fitc_intensities, opacity=0.8),
        ...         Layer(TRITC, tritc_intensities, transparent=False),
        ...     ]
        ... )
    """
    if background.ndim != 2:
        raise ValueError(f"Expected 2D background array, got shape {background.shape}")

    result = gray2rgb(background)

    for layer in layers:
        colormap = _make_colormap(layer.channel.color, layer.channel.name, layer.transparent)
        foreground_rgba = colorize(layer.intensities, colormap)
        foreground_rgb = foreground_rgba[..., :3]
        alpha = layer.opacity * foreground_rgba[..., 3:4]
        result = alpha_blend(result, foreground_rgb, alpha)

    return result


def alpha_blend(
    background: Float64Array,
    foreground: Float64Array,
    alpha: Float64Array,
) -> Float64Array:
    """Alpha-blend foreground onto background.

    Args:
        background: Background image with values in [0, 1].
        foreground: Foreground image with values in [0, 1].
        alpha: Alpha values in [0, 1]. Can be per-pixel (HxWx1) or scalar.

    Returns:
        Blended image with values clipped to [0, 1].
    """
    result = alpha * foreground + (1 - alpha) * background
    return np.clip(result, 0, 1)


def colorize(
    intensities: Float64Array,
    colormap: LinearSegmentedColormap,
) -> Float64Array:
    """Apply a colormap to a 2D intensity array.

    Args:
        intensities: 2D array of intensity values in [0, 1].
        colormap: Matplotlib colormap to apply.

    Returns:
        RGBA image (HxWx4 array).

    Raises:
        ValueError: If intensities is not a 2D array.
    """
    if intensities.ndim != 2:
        raise ValueError(
            f"Expected 2D array, but input has shape {intensities.shape} "
            f"with {intensities.ndim} dimensions."
        )

    norm = Normalize(vmin=0, vmax=1)
    mapper = ScalarMappable(norm=norm, cmap=colormap)
    return mapper.to_rgba(intensities).astype(np.float64)


def _make_colormap(color: str, name: str, transparent: bool) -> LinearSegmentedColormap:
    """Build a colormap for a single channel."""
    if transparent:
        return _semitransparent_colormap(color, name)
    return _opaque_colormap(color, name)


def _opaque_colormap(color: str, name: str) -> LinearSegmentedColormap:
    """Colormap from black to *color*."""
    return LinearSegmentedColormap.from_list(name, [(0, 0, 0, 1), color])


def _semitransparent_colormap(color: str, name: str) -> LinearSegmentedColormap:
    """Colormap from fully-transparent neutral gray to *color*.

    The zero-point anchors at a neutral gray (0.5, 0.5, 0.5) with full transparency rather
    than black (0, 0, 0). This was chosen empirically: on typical grayscale brightfield
    backgrounds, a gray anchor produces smoother blending and avoids the dark halos that a
    black anchor introduces around low-signal regions.
    """
    return LinearSegmentedColormap.from_list(name, [(0.5, 0.5, 0.5, 0), color])
