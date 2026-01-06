from __future__ import annotations

from arcadia_pycolor import HexCode
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage.color import gray2rgb

from .channels import Channel
from .typing import FloatArray


def create_overlay(
    background: FloatArray,
    foreground_rgba: FloatArray,
    opacity: float = 1,
) -> FloatArray:
    """Create an overlay by alpha blending foreground onto background.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        foreground_rgba: RGBA foreground image (HxWx4 array) with values in [0, 1].
        opacity: Global opacity multiplier for the foreground. Default is 1 (fully opaque).

    Returns:
        RGB image (HxWx3 array) with foreground alpha-blended onto background.
    """
    background_rgb = gray2rgb(background)
    foreground_rgb = foreground_rgba[..., :3]
    alpha = opacity * foreground_rgba[..., 3:4]
    return alpha_blend(background_rgb, foreground_rgb, alpha)


def alpha_blend(
    background: FloatArray,
    foreground: FloatArray,
    alpha: FloatArray,
) -> FloatArray:
    """Alpha blend foreground onto background.

    Args:
        background: Background image with values in [0, 1].
        foreground: Foreground image with values in [0, 1].
        alpha: Alpha channel values in [0, 1]. Can be per-pixel (HxWx1 array) or scalar.

    Returns:
        Blended image.
    """
    return alpha * foreground + (1 - alpha) * background


def colorize(
    intensities: FloatArray,
    colormap: LinearSegmentedColormap,
) -> FloatArray:
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
    return mapper.to_rgba(intensities)


def channel_to_opaque_colormap(channel: Channel) -> LinearSegmentedColormap:
    """Create an opaque colormap from a Channel's color.

    Args:
        channel: The Channel to create a colormap from.

    Returns:
        A LinearSegmentedColormap from black to the channel's color.

    Raises:
        ValueError: If the channel has no color defined.
    """
    if channel.color is None:
        raise ValueError(f"Channel '{channel.name}' has no color")
    return create_opaque_colormap(color=channel.color, name=channel.name)


def channel_to_semitransparent_colormap(channel: Channel) -> LinearSegmentedColormap:
    """Create a semi-transparent colormap from a Channel's color.

    Args:
        channel: The Channel to create a colormap from.

    Returns:
        A LinearSegmentedColormap from transparent white to the channel's color.

    Raises:
        ValueError: If the channel has no color defined.
    """
    if channel.color is None:
        raise ValueError(f"Channel '{channel.name}' has no color")
    return create_semitransparent_colormap(color=channel.color, name=channel.name)


def create_opaque_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a colormap from black to the given color."""
    colors = [
        (0, 0, 0, 1),
        color.hex_code,
    ]
    name = color.name if name is None else name
    colormap = LinearSegmentedColormap.from_list(name, colors)
    return colormap


def create_semitransparent_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a semi-transparent colormap for the given color."""
    colors = [
        (1, 1, 1, 0),
        color.hex_code,
    ]
    name = color.name if name is None else name
    colormap = LinearSegmentedColormap.from_list(name, colors)
    return colormap
