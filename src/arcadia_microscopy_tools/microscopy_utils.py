from __future__ import annotations
from dataclasses import dataclass
from enum import Flag, StrEnum, auto

from .channels import Channel


class DimensionFlags(Flag):
    """Bit flags for what dimensions are present."""

    SPATIAL_2D = auto()
    MULTICHANNEL = auto()
    TIME_SERIES = auto()
    Z_STACK = auto()
    SPECTRAL = auto()
    RGB = auto()
    MONTAGE = auto()

    @property
    def is_multichannel(self) -> bool:
        return bool(self & DimensionFlags.MULTICHANNEL)

    @property
    def is_timelapse(self) -> bool:
        return bool(self & DimensionFlags.TIME_SERIES)

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
class AcquisitionSettings:
    """"""

    exposure_time_ms: float | None = None
    zoom: float | None = None
    binning: str | None = None
    period_ms: float | None = None
    duration_s: float | None = None
    laser_power_pct: float | None = None


@dataclass
class MicroscopeSettings:
    """"""

    objective: str | None = None
    magnification: float | None = None
    numerical_aperture: float | None = None
    light_source: str | None = None


@dataclass
class ChannelMetadata:
    """
    Docstring for ChannelMetadata
    """

    channel: Channel
    resolution: PhysicalDimensions
    acquisition: AcquisitionSettings
    optics: MicroscopeSettings
