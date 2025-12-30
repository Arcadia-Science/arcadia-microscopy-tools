from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Flag, auto

from .channels import Channel


class DimensionFlags(Flag):
    """Bit flags for what dimensions are present."""

    SPATIAL_2D = auto()
    MULTICHANNEL = auto()
    TIMELAPSE = auto()
    Z_STACK = auto()
    SPECTRAL = auto()
    RGB = auto()
    MONTAGE = auto()

    @property
    def is_multichannel(self) -> bool:
        return bool(self & DimensionFlags.MULTICHANNEL)

    @property
    def is_timelapse(self) -> bool:
        return bool(self & DimensionFlags.TIMELAPSE)

    @property
    def is_zstack(self) -> bool:
        return bool(self & DimensionFlags.Z_STACK)

    @property
    def is_spectral(self) -> bool:
        return bool(self & DimensionFlags.SPECTRAL)

    @property
    def is_rgb(self) -> bool:
        return bool(self & DimensionFlags.RGB)

    @property
    def is_montage(self) -> bool:
        return bool(self & DimensionFlags.MONTAGE)


@dataclass
class PhysicalDimensions:
    """"""

    height_px: int
    width_px: int
    pixel_size_um: tuple[float, float]

    # Relevant for Z_STACK
    thickness_px: int | None = None
    z_step_size_um: float | None = None


@dataclass
class AcquisitionSettings:
    """"""

    exposure_time_ms: float
    zoom: float | None
    binning: str | None

    # Relevant for TIMELAPSE
    period_ms: float | None = None
    duration_s: float | None = None

    # Relevant for SPECTRAL
    min_wavelength_nm: float | None = None
    max_wavelength_nm: float | None = None
    min_wavenumber_cm1: float | None = None
    max_wavenumber_cm1: float | None = None


@dataclass
class MicroscopeSettings:
    """"""

    magnification: int
    numerical_aperture: float
    objective: str | None = None
    light_source: str | None = None
    laser_power_mw: float | None = None


@dataclass
class ChannelMetadata:
    """
    Docstring for ChannelMetadata
    """

    channel: Channel
    timestamp: datetime
    resolution: PhysicalDimensions
    acquisition: AcquisitionSettings
    optics: MicroscopeSettings
