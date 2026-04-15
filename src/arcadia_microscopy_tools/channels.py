"""Microscopy channel definitions and wavelength-to-color utilities."""

from __future__ import annotations
import re
from dataclasses import dataclass

import colour
import numpy as np

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


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


@dataclass(frozen=True)
class Channel:
    """A microscopy imaging channel.

    Attributes:
        name: Human-readable channel identifier (e.g. ``"DAPI"``).
        color: Hex color string used for visualization (e.g. ``"#0033FF"``).
        excitation_nm: Excitation wavelength in nanometers, if known.
        emission_nm: Emission wavelength in nanometers, if known.
    """

    name: str
    color: str
    excitation_nm: float | None = None
    emission_nm: float | None = None

    def __post_init__(self) -> None:
        if not _HEX_RE.match(self.color):
            raise ValueError(f"color must be a hex code like '#FF0000', got '{self.color}'")
        if self.excitation_nm is not None and self.excitation_nm <= 0:
            raise ValueError("excitation_nm must be positive")
        if self.emission_nm is not None and self.emission_nm <= 0:
            raise ValueError("emission_nm must be positive")

    @classmethod
    def from_wavelength(
        cls,
        wavelength_nm: float,
        *,
        name: str | None = None,
        is_excitation: bool = True,
    ) -> Channel:
        """Create a channel with a color derived from a visible wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers (360-780).
            name: Channel name. Defaults to ``"{wavelength}nm"``.
            is_excitation: If True (default), ``wavelength_nm`` is stored as excitation.
                Otherwise it is stored as emission.
        """
        hex_color = wavelength_to_hex(wavelength_nm)
        name = name or f"{wavelength_nm:.0f}nm"
        wl = round(wavelength_nm, 1)
        return cls(
            name=name,
            color=hex_color,
            excitation_nm=wl if is_excitation else None,
            emission_nm=wl if not is_excitation else None,
        )


# ── Predefined channels ─────────────────────────────────────────────────────

BRIGHTFIELD = Channel("BRIGHTFIELD", "#FFFFFF")
DIC = Channel("DIC", "#FFFFFF")
PHASE = Channel("PHASE", "#DDDDDD")
DAPI = Channel("DAPI", "#0033FF", excitation_nm=405, emission_nm=450)
FITC = Channel("FITC", "#07FF00", excitation_nm=488, emission_nm=512)
TRITC = Channel("TRITC", "#FFBF00", excitation_nm=561, emission_nm=595)
CY5 = Channel("CY5", "#A30000", excitation_nm=640, emission_nm=665)
SRS = Channel("SRS", "#E63535")
E_CARS = Channel("E-CARS", "#AB1299")
F_CARS = Channel("F-CARS", "#AB1299")
E_SHG = Channel("E-SHG", "#F29B4F")
F_SHG = Channel("F-SHG", "#F29B4F")

CHANNELS: dict[str, Channel] = {
    ch.name: ch
    for ch in [
        BRIGHTFIELD,
        DIC,
        PHASE,
        DAPI,
        FITC,
        TRITC,
        CY5,
        SRS,
        E_CARS,
        F_CARS,
        E_SHG,
        F_SHG,
    ]
}
