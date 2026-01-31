from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Flag, auto
from typing import TYPE_CHECKING

from .channels import Channel
from .typing import Float64Array

if TYPE_CHECKING:
    from dataclasses import Field
    from typing import Any


def dimension_field(dimension: DimensionFlags, default=None):
    """Create a field that's required for a specific dimension."""
    return field(default=default, metadata={"requires_dimension": dimension})


class DimensionValidatorMixin:
    """Mixin that provides dimension-based field validation for dataclasses."""

    if TYPE_CHECKING:
        __dataclass_fields__: dict[str, Field[Any]]

    def validate(self, dimensions: DimensionFlags) -> None:
        """Validate that required fields are present for the given dimensions."""
        for field_info in self.__dataclass_fields__.values():
            required_dimension = field_info.metadata.get("requires_dimension")
            if required_dimension and (dimensions & required_dimension):
                if getattr(self, field_info.name) is None:
                    raise ValueError(f"{field_info.name} is required for {required_dimension.name}")


class DimensionFlags(Flag):
    """Bit flags for what dimensions are present."""

    SPATIAL_2D = 0
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
class PhysicalDimensions(DimensionValidatorMixin):
    """Physical dimensions of the imaging volume."""

    height_px: int
    width_px: int
    pixel_size_um: float
    thickness_px: int | None = dimension_field(DimensionFlags.Z_STACK)
    z_step_size_um: float | None = dimension_field(DimensionFlags.Z_STACK)


@dataclass
class AcquisitionSettings(DimensionValidatorMixin):
    """Acquisition parameters for image capture."""

    exposure_time_ms: float
    zoom: float | None = None
    binning: str | None = None
    frame_intervals_ms: Float64Array | None = dimension_field(DimensionFlags.TIMELAPSE)
    wavelengths_nm: Float64Array | None = dimension_field(DimensionFlags.SPECTRAL)


@dataclass
class MicroscopeSettings:
    """Microscope optical configuration and settings."""

    magnification: int
    numerical_aperture: float
    objective: str | None = None
    light_source: str | None = None
    power_mw: float | None = None


@dataclass
class ChannelMetadata:
    """Metadata for a microscopy channel."""

    channel: Channel
    timestamp: datetime
    dimensions: DimensionFlags
    resolution: PhysicalDimensions
    acquisition: AcquisitionSettings
    optics: MicroscopeSettings

    def __post_init__(self):
        """Validate all sub-components against dimension flags."""
        self.resolution.validate(self.dimensions)
        self.acquisition.validate(self.dimensions)
