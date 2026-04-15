from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from arcadia_pycolor import HexCode
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage.color import gray2rgb

from .channels import Channel
from .typing import Float64Array


@dataclass
class Layer:
    """A single layer in a fluorescence overlay.

    Args:
        channel: Channel object containing color and metadata.
        intensities: 2D array of intensity values in [0, 1].
        opacity: Global opacity multiplier for this layer in [0, 1]. Default is 1 (fully opaque).
        transparent: If True (default), colormap goes from transparent to channel color.
            If False, colormap goes from black to channel color.
    """

    channel: Channel
    intensities: Float64Array
    opacity: float = 1.0
    transparent: bool = True

    def __post_init__(self) -> None:
        if self.channel.color is None:
            raise ValueError(f"Channel '{self.channel.name}' has no color defined")
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
    """Create a fluorescence overlay.

    All channels are blended with the same opacity and transparency settings.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        channel_intensities: Dict mapping Channel objects to their intensity arrays
            (2D, values in [0, 1]).
        opacity: Global opacity multiplier for all channels. Default is 1 (fully opaque).
        transparent: If True (default), all colormaps go from transparent to channel color.
            If False, all colormaps go from black to channel color.

    Returns:
        RGB image (HxWx3 array) with all channels alpha-blended onto background.

    Example:
        >>> # Simple overlay with default settings
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
    """Create an overlay by sequentially blending multiple channels onto a background.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        layers: List of Layer objects to overlay in sequence.

    Returns:
        RGB image (HxWx3 array) with all layers alpha-blended onto background.

    Raises:
        ValueError: If background is not a 2D array.

    Example:
        >>> # Fine-grained control over each layer
        >>> overlay = create_sequential_overlay(
        ...     background=brightfield,
        ...     layers=[
        ...         Layer(DAPI, dapi_intensities),
        ...         Layer(FITC, fitc_intensities, opacity=0.8),
        ...         Layer(TRITC, tritc_intensities, transparent=False),  # Opaque colormap
        ...     ]
        ... )
    """
    if background.ndim != 2:
        raise ValueError(f"Expected 2D background array, got shape {background.shape}")

    result = gray2rgb(background)

    for layer in layers:
        colormap = _channel_to_colormap(layer.channel, transparent=layer.transparent)
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
    """Alpha blend foreground onto background.

    Args:
        background: Background image with values in [0, 1].
        foreground: Foreground image with values in [0, 1].
        alpha: Alpha channel values in [0, 1]. Can be per-pixel (HxWx1 array) or scalar.

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
        RGBA image (HxWx4 array) with the colormap applied.

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


def _channel_to_colormap(
    channel: Channel,
    transparent: bool = True,
) -> LinearSegmentedColormap:
    """Convert a Channel to a matplotlib colormap.

    Args:
        channel: Channel object containing color information. Must have a color defined.
        transparent: If True (default), creates a colormap from transparent to channel color.
            If False, creates a colormap from black to channel color.

    Returns:
        LinearSegmentedColormap suitable for visualizing the channel.

    Raises:
        ValueError: If the channel has no color defined.
    """
    if channel.color is None:
        raise ValueError(f"Channel '{channel.name}' has no color defined")
    if transparent:
        return _create_semitransparent_colormap(color=channel.color, name=channel.name)
    else:
        return _create_opaque_colormap(color=channel.color, name=channel.name)


def _create_opaque_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a colormap from black to the given color."""
    colors = [
        (0, 0, 0, 1),
        color.hex_code,
    ]
    name = color.name if name is None else name
    return LinearSegmentedColormap.from_list(name, colors)


def _create_semitransparent_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a semi-transparent colormap for the given color.

    The zero-point anchors at a neutral gray (0.5, 0.5, 0.5) with full transparency rather
    than black (0, 0, 0). This was chosen empirically: on typical grayscale brightfield
    backgrounds, a gray anchor produces smoother blending and avoids the dark halos that a
    black anchor introduces around low-signal regions.
    """
    colors = [
        (0.5, 0.5, 0.5, 0),
        color.hex_code,
    ]
    name = color.name if name is None else name
    return LinearSegmentedColormap.from_list(name, colors)
