"""Visualization helpers for mapping channels to display colors.

Display color is a rendering concern, not part of a channel's identity, so it
lives here rather than on :class:`~arcadia_microscopy_tools.channels.Channel`.
"""

from __future__ import annotations
import re

import colour
import numpy as np

from .channels import Channel, Modality, SpectralChannel

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")

# Fallback for signals outside the visible range (e.g. NIR SRS emission).
_NIR_FALLBACK = "#8B0000"

# Fixed display colors for label-free transmitted-contrast modalities.
_MODALITY_COLORS: dict[Modality, str] = {
    Modality.BRIGHTFIELD: "#FFFFFF",
    Modality.DIC: "#FFFFFF",
    Modality.PHASE: "#DDDDDD",
}


def wavelength_to_hex(wavelength_nm: float) -> str:
    """Convert a visible-spectrum wavelength to a hex color string.

    Args:
        wavelength_nm: Wavelength in nanometers (360-780).

    Returns:
        Hex color string, e.g. ``"#1A2BFF"``.

    Raises:
        ValueError: If wavelength is outside the visible range.
    """
    if not 360 <= wavelength_nm <= 780:
        raise ValueError(
            f"Wavelength must be in the visible range (360-780 nm), got {wavelength_nm} nm"
        )
    xyz = colour.wavelength_to_XYZ(wavelength_nm)
    rgb = np.clip(colour.XYZ_to_sRGB(xyz), 0, 1)
    r, g, b = (rgb * 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def channel_color(channel: Channel) -> str:
    """Return a display color for a channel.

    Label-free modalities use fixed neutral colors. Spectral channels derive
    their color from the emission wavelength (falling back to excitation), or to
    a dark red when no wavelength is in the visible range.

    Args:
        channel: The channel to assign a display color to.

    Returns:
        Hex color string, e.g. ``"#07FF00"``.
    """
    if channel.modality in _MODALITY_COLORS:
        return _MODALITY_COLORS[channel.modality]

    wavelength_nm: float | None = None
    if isinstance(channel, SpectralChannel):
        wavelength_nm = channel.emission_nm or channel.excitation_nm

    if wavelength_nm is not None and 360 <= wavelength_nm <= 780:
        return wavelength_to_hex(wavelength_nm)

    return _NIR_FALLBACK
