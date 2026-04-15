from __future__ import annotations
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .channels import Channel
from .typing import Float64Array


class BlendMode(Enum):
    """How a foreground layer is composited onto the canvas.

    ALPHA:
        Standard Porter-Duff "over" compositing.  The foreground replaces the
        background in proportion to alpha.  Layer order matters.

    ADDITIVE:
        The foreground contribution is *added* to the background, then
        clipped.  This is the physically-motivated model for fluorescence:
        each fluorophore contributes light independently, so contributions
        from overlapping channels accumulate.  Layer order does not matter.
    """

    ALPHA = "alpha"
    ADDITIVE = "additive"


@dataclass
class Layer:
    """A single layer in a fluorescence overlay.

    Args:
        channel: Channel providing color and identity.
        intensities: 2D array of intensity values in [0, 1].
        opacity: Global opacity multiplier in [0, 1]. Default is 1 (fully opaque).
        zero_transparent: If True (default), the colormap fades from fully
            transparent at zero intensity to the channel color at full intensity.
            If False, the colormap fades from black to the channel color (no
            transparency is applied; useful when there is no meaningful
            background to show through).
        blend_mode: How this layer is composited onto the canvas.
            Default is ``ALPHA``.
    """

    channel: Channel
    intensities: Float64Array
    opacity: float = 1.0
    zero_transparent: bool = True
    blend_mode: BlendMode = BlendMode.ALPHA

    def __post_init__(self) -> None:
        if self.intensities.ndim != 2:
            raise ValueError(f"Expected 2D intensities array, got shape {self.intensities.shape}")
        if not 0 <= self.opacity <= 1:
            raise ValueError(f"Opacity must be in [0, 1], got {self.opacity}")

        lo, hi = float(self.intensities.min()), float(self.intensities.max())
        if lo < 0.0 or hi > 1.0:
            warnings.warn(
                f"Layer '{self.channel.name}' has intensity values outside [0, 1] "
                f"(min={lo:.4g}, max={hi:.4g}). Values will be clipped, which "
                f"may indicate missing normalization.",
                stacklevel=2,
            )
            self.intensities = np.clip(self.intensities, 0.0, 1.0)


def overlay_channels(
    background: Float64Array,
    channel_intensities: dict[Channel, Float64Array],
    *,
    opacity: float = 1.0,
    zero_transparent: bool = True,
    blend_mode: BlendMode = BlendMode.ALPHA,
) -> Float64Array:
    """Create a fluorescence overlay with uniform settings for all channels.

    This is the high-level convenience function.  For per-layer control over
    opacity, transparency, or blend mode, use :func:`create_overlay` directly.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        channel_intensities: Dict mapping Channel objects to their 2D intensity
            arrays (values in [0, 1]).
        opacity: Global opacity multiplier for all channels. Default is 1.
        zero_transparent: If True (default), all colormaps fade from transparent
            to channel color.  If False, colormaps fade from black.
        blend_mode: Compositing mode for all channels.  Default is
            ``BlendMode.ALPHA``.

    Returns:
        RGB image (HxWx3 float64 array) with all channels composited onto
        the background.

    Example:
        >>> overlay = overlay_channels(
        ...     background=brightfield,
        ...     channel_intensities={
        ...         DAPI: dapi_intensities,
        ...         FITC: fitc_intensities,
        ...         TRITC: tritc_intensities,
        ...     },
        ... )
    """
    layers = [
        Layer(channel, intensities, opacity, zero_transparent, blend_mode)
        for channel, intensities in channel_intensities.items()
    ]
    return create_overlay(background, layers)


def create_overlay(
    background: Float64Array,
    layers: list[Layer],
) -> Float64Array:
    """Create an overlay by compositing layers onto a background.

    Args:
        background: 2D grayscale background image with values in [0, 1].
        layers: List of Layer objects to composite.

    Returns:
        RGB image (HxWx3 float64 array) with all layers composited onto
        the background.

    Raises:
        ValueError: If the background is not 2D, or if any layer's spatial
            dimensions do not match the background.

    Example:
        >>> overlay = create_overlay(
        ...     background=brightfield,
        ...     layers=[
        ...         Layer(DAPI, dapi_intensities),
        ...         Layer(FITC, fitc_intensities, opacity=0.8),
        ...         Layer(TRITC, tritc_intensities, blend_mode=BlendMode.ALPHA),
        ...     ],
        ... )
    """
    if background.ndim != 2:
        raise ValueError(f"Expected 2D background array, got shape {background.shape}")

    lo, hi = float(background.min()), float(background.max())
    if lo < 0.0 or hi > 1.0:
        warnings.warn(
            f"Background has values outside [0, 1] (min={lo:.4g}, max={hi:.4g}). "
            f"Values will be clipped, which may indicate missing normalization.",
            stacklevel=2,
        )
        background = np.clip(background, 0.0, 1.0)

    canvas = _gray_to_rgb(background)

    for layer in layers:
        if layer.intensities.shape != background.shape:
            raise ValueError(
                f"Layer '{layer.channel.name}' has shape "
                f"{layer.intensities.shape}, but background has shape "
                f"{background.shape}."
            )
        cmap = _build_colormap(layer.channel.color, layer.zero_transparent)
        rgba = cmap(layer.intensities)
        rgb = rgba[..., :3]
        alpha = layer.opacity * rgba[..., 3:4]
        canvas = _composite(canvas, rgb, alpha, layer.blend_mode)

    return canvas


def _composite(
    background: Float64Array,
    foreground: Float64Array,
    alpha: Float64Array,
    mode: BlendMode,
) -> Float64Array:
    """Composite *foreground* onto *background* using the given blend mode."""
    if mode is BlendMode.ADDITIVE:
        return _blend_additive(background, foreground, alpha)
    return _blend_alpha(background, foreground, alpha)


def _blend_alpha(
    background: Float64Array,
    foreground: Float64Array,
    alpha: Float64Array,
) -> Float64Array:
    """Porter-Duff 'over' compositing."""
    return np.clip(alpha * foreground + (1 - alpha) * background, 0.0, 1.0)


def _blend_additive(
    background: Float64Array,
    foreground: Float64Array,
    alpha: Float64Array,
) -> Float64Array:
    """Additive (screen-like) compositing -- contributions accumulate."""
    return np.clip(background + alpha * foreground, 0.0, 1.0)


@lru_cache(maxsize=64)
def _build_colormap(color: str, zero_transparent: bool) -> LinearSegmentedColormap:
    """Return a two-stop colormap for *color*, with LRU caching.

    When *zero_transparent* is True the zero-point is a fully-transparent
    neutral gray (0.5, 0.5, 0.5, 0).  The gray anchor was chosen empirically:
    on typical grayscale brightfield backgrounds it produces smoother blending
    and avoids the dark halos that a black anchor introduces around low-signal
    regions.

    When *zero_transparent* is False the zero-point is opaque black (0, 0, 0, 1),
    giving a classic LUT-style ramp.
    """
    if zero_transparent:
        stops = [(0.5, 0.5, 0.5, 0.0), color]
    else:
        stops = [(0.0, 0.0, 0.0, 1.0), color]
    return LinearSegmentedColormap.from_list(f"_chan_{color}", stops)


def _gray_to_rgb(image: Float64Array) -> Float64Array:
    """Broadcast a single-channel 2D image to (H, W, 3)."""
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)
