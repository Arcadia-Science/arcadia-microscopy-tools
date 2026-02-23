from __future__ import annotations
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import liffile
import numpy as np
from pydantic import BaseModel, computed_field

from .channels import BRIGHTFIELD, CARS, SHG, SRS, Channel
from .metadata_structures import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MeasuredDimensions,
    MicroscopeConfig,
    NominalDimensions,
)
from .microscopy import ImageMetadata
from .typing import Float64Array

_SI_UNITS: dict[str, float] = {
    "m": 1,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "s": 1,
    "ms": 1e-3,
    "us": 1e-6,
}

CRS_STOKES_WAVELENGTH_NM: float = 1031.7


def list_image_names(lif_path: Path) -> list[str]:
    """List all image names contained in a LIF file.

    Args:
        lif_path: Path to the Leica LIF file

    Returns:
        List of image names in the file
    """
    with liffile.LifFile(lif_path) as f:
        return [image.name for image in f.images]


def create_image_metadata_from_lif(
    lif_path: Path,
    image_name: str,
    channels: list[Channel] | None = None,
) -> ImageMetadata:
    """Create ImageMetadata from a Leica LIF file.

    Args:
        lif_path: Path to the Leica LIF file.
        image_name: Name of the specific image within the LIF file to extract.
        channels: Optional list of Channel objects to override automatic channel detection.
            If not provided, channels are inferred from the LIF file metadata.

    Returns:
        ImageMetadata with sizes and channel metadata for all channels.
    """
    parser = _LeicaMetadataParser(lif_path, image_name, channels)
    return parser.parse()


def calculate_raman_shift(
    pump_wavelength_nm: float | Float64Array,
    stokes_wavelength_nm: float | Float64Array = CRS_STOKES_WAVELENGTH_NM,
) -> float | Float64Array:
    """Calculate Raman shift from pump and Stokes wavelengths.

    Args:
        pump_wavelength_nm: Wavelength of the pump beam in nanometers.
            Can be a scalar or array of wavelengths.
        stokes_wavelength_nm: Wavelength of the Stokes beam in nanometers.
            Can be a scalar or array. Defaults to 1031.7 nm (CRS laser).

    Returns:
        Raman shift in wavenumbers (cm⁻¹). Returns array if either input is an array.
    """
    return (1 / pump_wavelength_nm - 1 / stokes_wavelength_nm) * 1e7


def calculate_antistokes_wavelength(
    pump_wavelength_nm: float | Float64Array,
    stokes_wavelength_nm: float | Float64Array = CRS_STOKES_WAVELENGTH_NM,
) -> float | Float64Array:
    """Calculate anti-Stokes wavelength from pump and Stokes wavelengths.

    Args:
        pump_wavelength_nm: Wavelength of the pump beam in nanometers.
            Can be a scalar or array of wavelengths.
        stokes_wavelength_nm: Wavelength of the Stokes beam in nanometers.
            Can be a scalar or array. Defaults to 1031.7 nm (CRS laser).

    Returns:
        Anti-Stokes wavelength in nanometers. Returns array if either input is an array.
    """
    return 1 / (2 / pump_wavelength_nm - 1 / stokes_wavelength_nm)


def _convert_units(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit not in _SI_UNITS:
        raise ValueError(f"Unknown unit {from_unit!r}")
    if to_unit not in _SI_UNITS:
        raise ValueError(f"Unknown unit {to_unit!r}")
    return value * _SI_UNITS[from_unit] / _SI_UNITS[to_unit]


def _get_required_attr(element: ET.Element, name: str) -> str:
    """Get a required attribute from an XML element."""
    value = element.get(name)
    if value is None:
        raise ValueError(f"Missing attribute {name!r} on <{element.tag}>")
    return value


class _LifChannel(BaseModel):
    """Recreated from liffile where it is not exposed and ignores properties."""

    data_type: int
    channel_tag: int
    resolution: int
    lut_name: str
    bytes_inc: int
    bit_inc: int
    min_value: float
    max_value: float
    unit: str
    name_of_measured_quantity: str
    properties: dict[str, str]

    model_config = {"frozen": True}

    @classmethod
    def from_xml(cls, element: ET.Element) -> _LifChannel:
        """Create from XML ChannelDescription element."""
        # Extract channel properties
        props: dict[str, str] = {}
        for prop in element.findall("ChannelProperty"):
            key_element = prop.find("Key")
            value_element = prop.find("Value")
            if key_element is None or value_element is None or key_element.text is None:
                continue
            props[key_element.text] = value_element.text or ""

        return cls(
            data_type=int(_get_required_attr(element, "DataType")),
            channel_tag=int(_get_required_attr(element, "ChannelTag")),
            resolution=int(_get_required_attr(element, "Resolution")),
            lut_name=_get_required_attr(element, "LUTName"),
            bytes_inc=int(_get_required_attr(element, "BytesInc")),
            bit_inc=int(_get_required_attr(element, "BitInc")),
            min_value=float(_get_required_attr(element, "Min")),
            max_value=float(_get_required_attr(element, "Max")),
            unit=element.get("Unit", ""),
            name_of_measured_quantity=element.get("NameOfMeasuredQuantity", ""),
            properties=props,
        )


class _LifDimension(BaseModel):
    """Recreated from liffile where it is not exposed."""

    dim_id: int
    number_of_elements: int
    origin: float
    length: float
    unit: str
    bit_inc: int
    bytes_inc: int

    model_config = {"frozen": True}

    @computed_field
    @property
    def step(self) -> float:
        """Calculate step size for this dimension."""
        return self.length / self.number_of_elements

    @classmethod
    def from_xml(cls, element: ET.Element) -> _LifDimension:
        """Create from XML DimensionDescription element."""
        return cls(
            dim_id=int(_get_required_attr(element, "DimID")),
            number_of_elements=int(_get_required_attr(element, "NumberOfElements")),
            origin=float(_get_required_attr(element, "Origin")),
            length=float(_get_required_attr(element, "Length")),
            unit=_get_required_attr(element, "Unit"),
            bit_inc=int(_get_required_attr(element, "BitInc")),
            bytes_inc=int(_get_required_attr(element, "BytesInc")),
        )


class _ImageDescription(BaseModel):
    """Container for LIF image description metadata including channels and dimensions."""

    lif_channels: list[_LifChannel]
    lif_dimensions: list[_LifDimension]

    model_config = {"frozen": True}


class _PowerState(str, Enum):
    """Laser power state enumeration."""

    ON = "On"
    OFF = "Off"


class _LightSourceType(int, Enum):
    """Light source type enumeration for different laser types."""

    DIODE = 1
    WLL = 4
    CRS = 6


class _LaserState(BaseModel):
    """Represents the state of a single laser in the system."""

    _LightSourceType: _LightSourceType
    LightSourceName: str
    WavelengthDouble: float
    _PowerState: _PowerState

    model_config = {"frozen": True, "extra": "ignore"}


class _LaserSystemState:
    """Collection of laser states for the entire laser system."""

    def __init__(self, lasers: list[_LaserState]) -> None:
        self.lasers = lasers

    @property
    def active_lasers(self) -> list[_LightSourceType]:
        """List of active laser types based on power state."""
        return [
            laser._LightSourceType for laser in self.lasers if laser._PowerState == _PowerState.ON
        ]

    def get_laser_by_type(self, laser_type: _LightSourceType) -> _LaserState:
        """Get laser state by light source type."""
        laser = next((laser for laser in self.lasers if laser._LightSourceType == laser_type), None)
        if laser is None:
            raise ValueError(f"No laser of type {laser_type!r} in laser system")
        return laser

    def get_laser_by_name(
        self, laser_name: Literal["UV Light", "SuperContVisible Light", "CARS Light (Attenuator)"]
    ) -> _LaserState:
        """Get laser state by light source name."""
        laser = next((laser for laser in self.lasers if laser.LightSourceName == laser_name), None)
        if laser is None:
            raise ValueError(f"No laser named {laser_name!r} in laser system")
        return laser


class _LaserValue(BaseModel):
    """Represents laser parameters at a specific step."""

    Step: int
    Wavelength: float
    Power: float
    FixedLinePower: float
    Temperature: float
    Humidity: float

    model_config = {"frozen": True}


class _LeicaMetadataParser:
    """Parser for extracting metadata from Leica LIF files."""

    # Set of detectors used for either the UV (405 nm) or WLL laser
    _FLUORESCENCE_DETECTORS = {"HyD S 1", "HyD S 2", "HyD X 3", "HyD R 4"}

    # Map of LIF dimension key → DimensionFlag for _get_dimension_flags
    _DIM_FLAG_MAP: dict[str, DimensionFlags] = {
        "T": DimensionFlags.TIMELAPSE,
        "Z": DimensionFlags.Z_STACK,
        "S": DimensionFlags.RGB,
        "λ": DimensionFlags.SPECTRAL,
        "Λ": DimensionFlags.SPECTRAL,
        "M": DimensionFlags.MONTAGE,
    }

    # Map of (detector_name, beam_route) to Channel for automatic channel detection
    _CHANNEL_DETECTION_MAP = {
        ("F-SRS", None): SRS,  # expected beam_route is "10;0" but not checked
        ("HyD NDD 1", "20;21"): CARS,
        # TODO: figure out beam route for true SHG vs pseudo-SHG (brightfield)
        # true SHG uses CRS pump laser (Stokes off) while pseudo-SHG uses WLL
        ("F-SHG", None): BRIGHTFIELD,
        # ("F-SHG", None): SHG,
        ("E-SHG", None): SHG,
    }

    def __init__(
        self,
        lif_path: Path,
        image_name: str,
        channels: list[Channel] | None = None,
    ):
        self.lif_path = lif_path
        self.image_name = image_name
        self.channels = channels
        # Attributes populated during parse()
        self._lif: liffile.LifFile
        self.image: Any
        self.sizes: dict[str, int]
        self.dimensions: DimensionFlags
        self.timestamp: datetime
        self.image_description: _ImageDescription
        self.laser_system_state: _LaserSystemState

    def parse(self) -> ImageMetadata:
        """Parse the LIF file and extract all metadata for the specified image."""
        with liffile.LifFile(self.lif_path) as self._lif:
            self.image = self._lif.images[self.image_name]

            # Validate critical metadata exists
            if not hasattr(self.image, "attrs"):
                raise ValueError(
                    f"Missing attrs metadata for image '{self.image_name}' in {self.lif_path}"
                )

            self.sizes = self.image.sizes
            self.dimensions = self._get_dimension_flags()
            self.timestamp = self._parse_timestamp()

            # Parse image description
            self.image_description = self._parse_image_description()

            # Parse laser system state
            self.laser_system_state = self._parse_laser_array_data()

            # Parse image-level metadata once, shared across all channels
            resolution = self._parse_nominal_dimensions()
            measured = self._parse_measured_dimensions()
            acquisition = self._parse_acquisition_settings()
            optics = self._parse_microscope_settings()

            channel_metadata_list = self._parse_all_channels(
                resolution, measured, acquisition, optics
            )
            return ImageMetadata(self.sizes, channel_metadata_list)

    def _parse_image_description(self) -> _ImageDescription:
        """Parse the _ImageDescription XML element into structured data.

        Returns:
            _ImageDescription containing channels and dimensions
        """
        # Find the _ImageDescription XML element
        image_description_element = self.image.xml_element.find("./Data/Image/_ImageDescription")
        if image_description_element is None:
            raise ValueError(
                f"Missing image description metadata for image '{self.image_name}' "
                f"in {self.lif_path}"
            )

        channels_element = image_description_element.find("Channels")
        dimensions_element = image_description_element.find("Dimensions")
        if channels_element is None or dimensions_element is None:
            raise ValueError("Expected <Channels> and <Dimensions> under <_ImageDescription>")

        lif_channels = [
            _LifChannel.from_xml(e) for e in channels_element.findall("ChannelDescription")
        ]
        lif_dimensions = [
            _LifDimension.from_xml(e) for e in dimensions_element.findall("DimensionDescription")
        ]

        return _ImageDescription(lif_channels=lif_channels, lif_dimensions=lif_dimensions)

    def _parse_laser_array_data(self) -> _LaserSystemState:
        """Parse laser system states from hardware settings."""

        laser_array_data = (
            self.image.attrs.get("HardwareSetting", {})
            .get("ATLConfocalSettingDefinition", {})
            .get("LaserArray", {})
            .get("Laser", {})
        )
        # Normalize to list: XML parsers may return a dict when there is only one element
        if isinstance(laser_array_data, dict):
            laser_array_data = [laser_array_data]
        return _LaserSystemState(
            lasers=[_LaserState(**laser_data) for laser_data in laser_array_data]
        )

    def _parse_all_channels(
        self,
        resolution: NominalDimensions,
        measured: MeasuredDimensions,
        acquisition: AcquisitionSettings,
        optics: MicroscopeConfig,
    ) -> list[ChannelMetadata]:
        """Parse metadata for all channels in the LIF image."""
        # Validate channels list length if provided
        num_channels = len(self.image_description.lif_channels)
        if self.channels is not None and len(self.channels) != num_channels:
            raise ValueError(
                f"Expected {num_channels} channels but got {len(self.channels)} in channels list"
            )

        return [
            self._parse_channel_metadata(
                lif_channel,
                self.channels[i] if self.channels else None,
                resolution,
                measured,
                acquisition,
                optics,
            )
            for i, lif_channel in enumerate(self.image_description.lif_channels)
        ]

    def _parse_channel_metadata(
        self,
        lif_channel: _LifChannel,
        channel: Channel | None,
        resolution: NominalDimensions,
        measured: MeasuredDimensions,
        acquisition: AcquisitionSettings,
        optics: MicroscopeConfig,
    ) -> ChannelMetadata:
        """Parse metadata for a specific channel."""
        if channel is None:
            channel = self._infer_channel(lif_channel)

        return ChannelMetadata(
            channel=channel,
            timestamp=self.timestamp,
            dimensions=self.dimensions,
            resolution=resolution,
            measured=measured,
            acquisition=acquisition,
            optics=optics,
        )

    def _infer_channel(self, lif_channel: _LifChannel) -> Channel:
        """Infer channel from LIF metadata using laser state and detector configuration.

        Channel inference is challenging due to limitations in LIF metadata structure:
        - Fluorescence detectors (HyD) don't explicitly indicate which laser was used,
          requiring heuristics to choose between WLL and DIODE lasers
        - Beam route information is inconsistent across detector types and may be
          missing or unchecked for some configurations
        - Some detectors map to different modalities depending on laser settings
          (e.g., F-SHG can be true SHG with CRS or pseudo-SHG/brightfield with WLL)
        - When multiple lasers are active, detector name and beam route are used to
          disambiguate, but this mapping is crude
        """
        active_lasers = self.laser_system_state.active_lasers
        if not active_lasers:
            raise ValueError(f"No active laser for '{self.image_name}' in {self.lif_path}")

        if len(active_lasers) == 1 and active_lasers[0] in (
            _LightSourceType.DIODE,
            _LightSourceType.WLL,
        ):
            active_laser_state = self.laser_system_state.get_laser_by_type(active_lasers[0])
            return self._infer_channel_from_laser_state(active_laser_state)

        return self._infer_channel_from_detector(lif_channel, active_lasers)

    def _infer_channel_from_laser_state(self, laser_state: _LaserState) -> Channel:
        """Infer channel from laser state using excitation wavelength."""
        if laser_state._LightSourceType == _LightSourceType.CRS:
            raise ValueError("Cannot infer channel from CRS laser")

        # Can reasonably infer channel only in the case where either the UV or WLL laser is ON
        excitation_wavelength_nm = self._extract_wavelength_value(laser_state.WavelengthDouble)
        try:
            return Channel.from_excitation_wavelength(
                excitation_wavelength_nm, name=laser_state._LightSourceType.name
            )
        except ValueError:
            warnings.warn(
                f"Parsed excitation wavelength {excitation_wavelength_nm} nm outside accepted "
                "range for Channel inference. Pass a Channel instance to prevent this warning.",
                stacklevel=2,
            )
            return Channel(name=laser_state._LightSourceType.name)

    def _infer_channel_from_detector(
        self,
        lif_channel: _LifChannel,
        active_lasers: list[_LightSourceType],
    ) -> Channel:
        """Infer channel from detector name and beam route.

        Args:
            lif_channel: The LIF channel description
            active_lasers: List of active laser types

        Returns:
            Channel inferred from detector configuration
        """
        detector_name = lif_channel.properties.get("DetectorName")
        beam_route = lif_channel.properties.get("BeamRoute")

        if detector_name in self._FLUORESCENCE_DETECTORS:
            # TODO: this is a crude assumption for WLL over DIODE
            laser_type = (
                _LightSourceType.WLL
                if _LightSourceType.WLL in active_lasers
                else _LightSourceType.DIODE
            )
            laser_state = self.laser_system_state.get_laser_by_type(laser_type)
            return self._infer_channel_from_laser_state(laser_state)

        # Try exact match with beam route, then fall back to match without beam route
        channel = self._CHANNEL_DETECTION_MAP.get(
            (detector_name, beam_route)
        ) or self._CHANNEL_DETECTION_MAP.get((detector_name, None))

        if channel is None:
            raise ValueError(
                f"Could not determine channel from DetectorName: {detector_name}, "
                f"BeamRoute: {beam_route}. Please provide channels list explicitly."
            )

        # For CARS and SRS, calculate wavelengths from CRS laser
        if channel in (CARS, SRS):
            laser_state = self.laser_system_state.get_laser_by_type(_LightSourceType.CRS)
            pump_wavelength_nm = self._extract_wavelength_value(laser_state.WavelengthDouble)

            if channel == CARS:
                # CARS detects anti-Stokes wavelength
                emission_nm = float(
                    calculate_antistokes_wavelength(pump_wavelength_nm, CRS_STOKES_WAVELENGTH_NM)
                )
            else:  # SRS
                # SRS is loss-based, emission wavelength equals excitation
                emission_nm = pump_wavelength_nm

            return Channel(
                name=channel.name,
                excitation_nm=round(pump_wavelength_nm, 1),
                emission_nm=round(emission_nm, 1),
                color=channel.color,
            )

        return channel

    def _get_dimension_flags(self) -> DimensionFlags:
        """Determine dimension flags from LIF file for a single channel.

        LIF images may have the following dimensions in almost any order:
            - "H": TCSPC histogram
            - "S": Sample/color component
            - "C": Channel
            - "X": Width
            - "Y": Height
            - "Z": Depth
            - "T": Time
            - "λ": Emission wavelength
            - "A": Rotation
            - "N": XT slices
            - "Q": T slices
            - "Λ": Excitation wavelength
            - "M": Mosaic ("S" in LAS X)
            - "L": Loop

        See also:
            - https://github.com/cgohlke/liffile/blob/main/liffile/liffile.py#L869
        """
        result = DimensionFlags(0)
        for key, flag in self._DIM_FLAG_MAP.items():
            if self.sizes.get(key, 0) > 1:
                result |= flag
        return result

    def _parse_timestamp(self) -> datetime:
        """Parse timestamp from LIF metadata."""
        try:
            return self._lif.images[self.image_name].timestamps[0]
        except IndexError:
            warnings.warn(
                f"Could not parse timestamp for image '{self.image_name}' in {self.lif_path}. "
                "Let's pretend it happened during the moon landing. Image could be corrupted.",
                stacklevel=2,
            )
            return datetime(1969, 7, 20, 20, 17)

    @property
    def _confocal_settings(self) -> dict:
        """Get ATLConfocalSettingDefinition from hardware settings."""
        return self.image.attrs.get("HardwareSetting", {}).get("ATLConfocalSettingDefinition", {})

    def _parse_nominal_dimensions(self) -> NominalDimensions:
        """Parse nominal dimensions from LIF metadata.

        Dimension ID legend:
            "X": dim_id = 1
            "Y": dim_id = 2
            "Z": dim_id = 3
            "T": dim_id = 4
            "λ": dim_id = 5
            ... ?
            "Λ": dim_id = 9
            "M": dim_id = 10
        """

        # X and Y dimensions
        x_dim = self._find_dimension(1)
        y_dim = self._find_dimension(2)
        x_step_um = _convert_units(x_dim.step, x_dim.unit, "um")
        y_step_um = _convert_units(y_dim.step, y_dim.unit, "um")
        if abs(x_step_um - y_step_um) / x_step_um > 0.01:
            warnings.warn(
                f"X ({x_step_um:.4f} µm) and Y ({y_step_um:.4f} µm) pixel steps differ by more "
                "than 1%; using average for xy_step_um.",
                stacklevel=2,
            )
        xy_step_um = (x_step_um + y_step_um) / 2

        # Optional Z dimension
        z_size_px, z_step_um = None, None
        if self.dimensions.is_zstack:
            z_dim = self._find_dimension(3)
            z_size_px = z_dim.number_of_elements
            z_step_um = _convert_units(z_dim.step, z_dim.unit, "um")

        # Optional T dimension
        t_size_px, t_step_ms = None, None
        if self.dimensions.is_timelapse:
            t_dim = self._find_dimension(4)
            t_size_px = t_dim.number_of_elements
            t_step_ms = _convert_units(t_dim.step, t_dim.unit, "ms")

        # Optional wavelength dimension (try Λ then λ)
        w_size_px, w_step_nm = None, None
        if self.dimensions.is_spectral:
            for dim_id, size_key in [(9, "Λ"), (5, "λ")]:
                if size_key in self.sizes and self.sizes[size_key] > 1:
                    w_dim = self._find_dimension(dim_id)
                    w_size_px = w_dim.number_of_elements
                    w_step_nm = _convert_units(w_dim.step, w_dim.unit, "nm")
                    break

        return NominalDimensions(
            x_size_px=x_dim.number_of_elements,
            y_size_px=y_dim.number_of_elements,
            xy_step_um=xy_step_um,
            z_size_px=z_size_px,
            z_step_um=z_step_um,
            t_size_px=t_size_px,
            t_step_ms=t_step_ms,
            w_size_px=w_size_px,
            w_step_nm=w_step_nm,
        )

    def _find_dimension(self, dim_id: int) -> _LifDimension:
        """Find a _LifDimension by its ID."""
        dimension = next(
            (d for d in self.image_description.lif_dimensions if d.dim_id == dim_id), None
        )
        if dimension is None:
            raise ValueError(f"Missing dimension (dim_id={dim_id}) in LIF metadata")
        return dimension

    def _parse_measured_dimensions(self) -> MeasuredDimensions:
        """Parse measured dimension values from LIF metadata.

        Coords are created in liffile.py from properties in LifDimension, hence they are not
        truly measured. Idk the real source of measured coordinates (or if there is one) for
        Z and T dimensions.

        See also:
            - https://github.com/cgohlke/liffile/blob/main/liffile/liffile.py#L1298
        """

        z_values_um = None
        if self.dimensions.is_zstack:
            z_dim = self._find_dimension(3)
            to_um = _convert_units(1, z_dim.unit, "um")
            z_values_um = to_um * self.image.coords["Z"]

        t_values_ms = None
        if self.dimensions.is_timelapse:
            t_dim = self._find_dimension(4)
            to_ms = _convert_units(1, t_dim.unit, "ms")
            t_values_ms = to_ms * self.image.coords["T"]

        w_values_nm = None
        if self.dimensions.is_spectral:
            laser_values_data = (
                self.image.attrs.get("_LaserValues", {})
                .get("Laser", {})
                .get("StagePosition", {})
                .get("_LaserValues", {})
            )
            # Normalize to list: XML parsers may return a dict when there is only one element
            if isinstance(laser_values_data, dict):
                laser_values_data = [laser_values_data]
            w_values_nm = np.array([_LaserValue(**item).Wavelength for item in laser_values_data])

        return MeasuredDimensions(
            z_values_um=z_values_um,
            t_values_ms=t_values_ms,
            w_values_nm=w_values_nm,
        )

    def _parse_acquisition_settings(self) -> AcquisitionSettings:
        """Parse acquisition settings from LIF metadata."""
        microscope_data = self._confocal_settings

        zoom = float(microscope_data.get("Zoom", np.nan))
        pixel_dwell_time_s = float(microscope_data.get("PixelDwellTime", np.nan))
        line_scan_speed_hz = float(microscope_data.get("ScanSpeed", np.nan))
        line_averaging = int(microscope_data.get("LineAverage", 1))
        line_accumulation = int(microscope_data.get("Line_Accumulation", 1))
        frame_averaging = int(microscope_data.get("FrameAverage", 1))
        frame_accumulation = int(microscope_data.get("FrameAccumulation", 1))

        # Convert pixel dwell time from seconds to microseconds
        pixel_dwell_time_us = 1e6 * pixel_dwell_time_s

        # Calculate total exposure time, accounting for all averaging and accumulation passes
        exposure_time_ms = (
            pixel_dwell_time_s
            * self.sizes["X"]
            * self.sizes["Y"]
            * line_averaging
            * line_accumulation
            * frame_averaging
            * frame_accumulation
            * 1e3
        )

        return AcquisitionSettings(
            exposure_time_ms=exposure_time_ms,
            zoom=zoom,
            binning=None,
            pixel_dwell_time_us=pixel_dwell_time_us,
            line_scan_speed_hz=line_scan_speed_hz,
            line_averaging=line_averaging,
            line_accumulation=line_accumulation,
            frame_averaging=frame_averaging,
            frame_accumulation=frame_accumulation,
        )

    def _parse_microscope_settings(self) -> MicroscopeConfig:
        """Parse microscope settings from LIF metadata."""
        microscope_data = self._confocal_settings

        magnification = int(microscope_data.get("Magnification", 0))
        numerical_aperture = float(microscope_data.get("NumericalAperture", np.nan))
        objective = microscope_data.get("ObjectiveName", "").strip()

        return MicroscopeConfig(
            magnification=magnification,
            numerical_aperture=numerical_aperture,
            objective=objective,
            light_source=None,
            power_mw=None,
        )

    @staticmethod
    def _extract_wavelength_value(value: str | int | float) -> float:
        """Extract a wavelength value and convert to nanometers if needed.

        Args:
            value: The value which might contain wavelength information

        Returns:
            Wavelength in nanometers

        Raises:
            ValueError: If value cannot be parsed as a wavelength
        """
        try:
            wavelength = float(value)
            # Convert to nm if it looks like it's in meters (< 1e-3, i.e. sub-millimeter)
            return wavelength * 1e9 if wavelength < 1e-3 else wavelength
        except (ValueError, TypeError) as ex:
            raise ValueError(f"Cannot determine wavelength from {value}") from ex
