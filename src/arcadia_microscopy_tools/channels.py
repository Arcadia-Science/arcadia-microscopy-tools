from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import colour
import numpy as np
from arcadia_pycolor import HexCode


@dataclass(frozen=True)
class Channel:
    """Represents a microscopy imaging channel with optical properties."""

    name: str
    excitation_nm: int | None = None
    emission_nm: int | None = None
    color: HexCode | None = None

    # Class-level registry of predefined channels
    registry: ClassVar[dict[str, Channel]] = {}

    def __post_init__(self):
        """Validate channel properties."""
        if self.excitation_nm is not None and self.excitation_nm <= 0:
            raise ValueError("Excitation wavelength must be positive")
        if self.emission_nm is not None and self.emission_nm <= 0:
            raise ValueError("Emission wavelength must be positive")

    @classmethod
    def register(cls, channel: Channel) -> Channel:
        """Register a predefined channel for easy lookup."""
        cls.registry[channel.name.upper()] = channel
        return channel

    @classmethod
    def get(cls, name: str) -> Channel | None:
        """Retrieve a predefined channel by name."""
        return cls.registry.get(name.upper())

    @classmethod
    def from_optical_config_name(cls, optical_config: str) -> Channel:
        """Get the Channel from the optical configuration name.

        Matches optical_config against registered Channel names by exact match or partial match.
        Case-insensitive matching is used. For partial matches, the longest matching channel
        name is returned to avoid ambiguity.

        Args:
            optical_config: The name of the optical configuration.

        Returns:
            Channel: The corresponding Channel object.

        Raises:
            ValueError: If the optical configuration is not recognized.
        """
        optical_config_upper = optical_config.upper()

        # Try exact match first
        if optical_config_upper in cls.registry:
            return cls.registry[optical_config_upper]

        # Handle special cases/known aliases
        if "mono" in optical_config.lower():
            return BRIGHTFIELD
        if "gfp" in optical_config.lower():
            return FITC

        # Try partial match (channel name appears in optical_config)
        # Return longest match to handle cases like "FITC" vs "FITC-A"
        matches = [
            channel_name for channel_name in cls.registry if channel_name in optical_config_upper
        ]
        if matches:
            # Sort by length (longest first) and return the longest match
            longest_match = max(matches, key=len)
            return cls.registry[longest_match]

        raise ValueError(f"{optical_config} is not a known optical configuration.")

    @classmethod
    def from_emission_wavelength(
        cls,
        wavelength_nm: float,
        excitation_nm: int | None = None,
        name: str | None = None,
    ) -> Channel:
        """Create a channel from an emission wavelength with automatically generated color.

        Args:
            wavelength_nm: Emission wavelength in nanometers. Valid range is 360-780 nm.
            excitation_nm: Optional excitation wavelength in nanometers.
            name: Optional name for the channel. If not provided, defaults to "{wavelength}nm".

        Returns:
            Channel with the emission wavelength and corresponding color.
        """
        color = cls.wavelength_to_color(wavelength_nm, name)
        return cls(
            name=color.name,
            excitation_nm=excitation_nm,
            emission_nm=int(wavelength_nm),
            color=color,
        )

    @classmethod
    def from_excitation_wavelength(
        cls,
        wavelength_nm: float,
        emission_nm: int | None = None,
        name: str | None = None,
    ) -> Channel:
        """Create a channel from an excitation wavelength with automatically generated color.

        Args:
            wavelength_nm: Excitation wavelength in nanometers. Valid range is 360-780 nm.
            emission_nm: Optional emission wavelength in nanometers.
            name: Optional name for the channel. If not provided, defaults to "{wavelength}nm".

        Returns:
            Channel with the excitation wavelength and corresponding color.
        """
        color = cls.wavelength_to_color(wavelength_nm, name)
        return cls(
            name=color.name,
            excitation_nm=int(wavelength_nm),
            emission_nm=emission_nm,
            color=color,
        )

    @staticmethod
    def wavelength_to_color(wavelength_nm: float, name: str | None = None) -> HexCode:
        """Convert a wavelength (nm) to a color representation.

        Args:
            wavelength_nm: Wavelength in nanometers. Valid range is between 360 and 780 nm.
            name: Optional name for the color. If not provided, defaults to "{wavelength}nm".

        Returns:
            HexCode object representing the color corresponding to the wavelength.

        Raises:
            ValueError: If wavelength is outside the valid range (360-780 nm).
        """
        if not 360 <= wavelength_nm <= 780:
            raise ValueError(
                "Wavelength must be within the visible range (between 360 and 780 nm), got "
                f"{wavelength_nm} nm"
            )

        name = f"{wavelength_nm:.0f}nm" if name is None else name

        # Convert wavelength to RGB color space
        xyz = colour.wavelength_to_XYZ(wavelength_nm)
        rgb = colour.XYZ_to_sRGB(xyz)

        # Clip values to valid range [0, 1] and handle out-of-gamut colors
        rgb = np.clip(rgb, 0, 1)

        # Convert to 0-255 range
        rgb_255 = (rgb * 255).astype(int)

        # Convert to hex code
        hex_code = f"#{rgb_255[0]:02X}{rgb_255[1]:02X}{rgb_255[2]:02X}"
        return HexCode(name, hex_code=hex_code)


BRIGHTFIELD = Channel.register(
    Channel(name="BRIGHTFIELD", color=HexCode("brightfield", "#ffffff")),
)
DIC = Channel.register(
    Channel(name="DIC", color=HexCode("dic", "#ffffff")),
)
PHASE = Channel.register(
    Channel(name="PHASE", color=HexCode("phase", "#dddddd")),
)
DAPI = Channel.register(
    Channel(name="DAPI", excitation_nm=405, emission_nm=450, color=HexCode("dapi", "#0033ff"))
)
FITC = Channel.register(
    Channel(name="FITC", excitation_nm=488, emission_nm=512, color=HexCode("fitc", "#07ff00"))
)
TRITC = Channel.register(
    Channel(name="TRITC", excitation_nm=561, emission_nm=595, color=HexCode("tritc", "#ffbf00"))
)
CY5 = Channel.register(
    Channel(name="CY5", excitation_nm=640, emission_nm=665, color=HexCode("cy5", "#a30000"))
)
CARS = Channel.register(
    Channel(name="CARS", color=HexCode("cars", "#AB1299")),
)
SRS = Channel.register(
    Channel(name="SRS", color=HexCode("srs", "#e63535")),
)
