from __future__ import annotations
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

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
    stokes_wavelength_nm: float | Float64Array = 1031.7,
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
    stokes_wavelength_nm: float | Float64Array = 1031.7,
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


class ImageDescription(BaseModel):
    """Container for LIF image description metadata including channels and dimensions."""

    lif_channels: list[_LifChannel]
    lif_dimensions: list[_LifDimension]

    model_config = {"frozen": True}


class PowerState(str, Enum):
    """Laser power state enumeration."""

    ON = "On"
    OFF = "Off"


class LightSourceType(int, Enum):
    """Light source type enumeration for different laser types."""

    DIODE = 1
    WLL = 4
    CRS = 6


class BeamPositionInfo(BaseModel):
    """Beam position information from hardware settings."""

    BeamPositionLevel: int
    BeamPosition: int

    model_config = {"frozen": True}


class BeamRoute(BaseModel):
    """Beam routing configuration for laser optics."""

    BeamPosition: BeamPositionInfo | list[BeamPositionInfo]
    Version: int

    model_config = {"frozen": True}


class LaserLine(BaseModel):
    """Individual laser line activation state."""

    IsLineActive: int

    model_config = {"frozen": True}


class LaserState(BaseModel):
    """Represents the state of a single laser in the system."""

    # Required fields
    BeamRoute: BeamRoute
    Version: int
    LightSourceType: LightSourceType
    LaserName: str
    StedAlignFlag: int
    CanDoLinearOutputPower: int
    CanDoPulsing: int
    CanDoOutputPowerWatt: int
    HighPowerModeActive: int
    LightSourceName: str
    OutputPowerWatt: float
    Wavelength: float
    WavelengthDouble: float
    PowerState: PowerState
    CanDoChangeWavelength: int

    # Optional fields (only present in some lasers)
    OutputPowerPercentage: float | None = None
    LaserLines: list[LaserLine] | None = None
    ShutterState: int | None = None
    ConstantPowerMode: int | None = None
    TargetOutputPower: float | None = None
    TuningRangeMin: float | None = None
    TuningRangeMax: float | None = None
    PumpOutputPowerPercent: float | None = None
    PumpOutputPower: float | None = None
    PumpWavelength: float | None = None

    model_config = {"frozen": True}


class LaserSystemState:
    """Collection of laser states for the entire laser system."""

    def __init__(self, lasers: list[LaserState]) -> None:
        self.lasers = lasers

    @property
    def active_lasers(self) -> list[LightSourceType]:
        """List of active laser types based on power state."""
        return [laser.LightSourceType for laser in self.lasers if laser.PowerState == PowerState.ON]

    def get_laser_by_type(self, laser_type: LightSourceType) -> LaserState:
        """Get laser state by light source type."""
        return next(laser for laser in self.lasers if laser.LightSourceType == laser_type)

    def get_laser_by_name(
        self, laser_name: Literal["UV Light", "SuperContVisible Light", "CARS Light (Attenuator)"]
    ) -> LaserState:
        """Get laser state by light source name."""
        return next(laser for laser in self.lasers if laser.LightSourceName == laser_name)


class LaserValue(BaseModel):
    """Represents laser parameters at a specific step."""

    Step: int
    Wavelength: float
    Power: float
    FixedLinePower: float
    Temperature: float
    Humidity: float

    model_config = {"frozen": True}


class LaserValueCollection:
    """Wrapper for operating on multiple LaserValue objects."""

    def __init__(self, laser_values: list[LaserValue]):
        self.values = laser_values

    @property
    def wavelengths_nm(self) -> Float64Array:
        """Array of wavelengths in nanometers."""
        return np.array([lv.Wavelength for lv in self.values])

    @property
    def powers_mw(self) -> Float64Array:
        """Array of laser powers in milliwatts."""
        return np.array([lv.Power for lv in self.values])

    @property
    def temperatures_c(self) -> Float64Array:
        """Array of temperatures in Celsius."""
        return np.array([lv.Temperature for lv in self.values])

    @property
    def humidities_pct(self) -> Float64Array:
        """Array of humidity percentages."""
        return np.array([lv.Humidity for lv in self.values])


class _LeicaMetadataParser:
    """Parser for extracting metadata from Leica LIF files."""

    # Set of detectors used for either the UV (405 nm) or WLL laser
    _FLUORESCENCE_DETECTORS = {"HyD S 1", "HyD S 2", "HyD X 3", "HyD R 4"}

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
        self._lif: liffile.LifFile

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

            channel_metadata_list = self._parse_all_channels()
            return ImageMetadata(self.sizes, channel_metadata_list)

    def _parse_image_description(self) -> ImageDescription:
        """Parse the ImageDescription XML element into structured data.

        Returns:
            ImageDescription containing channels and dimensions
        """
        # Find the ImageDescription XML element
        image_description_element = self.image.xml_element.find("./Data/Image/ImageDescription")
        if image_description_element is None:
            raise ValueError(
                f"Missing image description metadata for image '{self.image_name}' "
                f"in {self.lif_path}"
            )

        channels_element = image_description_element.find("Channels")
        dimensions_element = image_description_element.find("Dimensions")
        if channels_element is None or dimensions_element is None:
            raise ValueError("Expected <Channels> and <Dimensions> under <ImageDescription>")

        lif_channels = self._parse_channels_from_xml(channels_element)
        lif_dimensions = self._parse_dimensions_from_xml(dimensions_element)

        return ImageDescription(lif_channels=lif_channels, lif_dimensions=lif_dimensions)

    def _parse_channels_from_xml(self, channels_element: ET.Element) -> list[_LifChannel]:
        """Parse channel descriptions from the <Channels> XML element."""
        return [
            _LifChannel.from_xml(element)
            for element in channels_element.findall("ChannelDescription")
        ]

    def _parse_dimensions_from_xml(self, dimensions_element: ET.Element) -> list[_LifDimension]:
        """Parse dimension descriptions from the <Dimensions> XML element."""
        return [
            _LifDimension.from_xml(element)
            for element in dimensions_element.findall("DimensionDescription")
        ]

    def _parse_laser_array_data(self) -> LaserSystemState:
        """Parse laser system states from hardware settings."""

        laser_array_data = (
            self.image.attrs.get("HardwareSetting", {})
            .get("ATLConfocalSettingDefinition", {})
            .get("LaserArray", {})
            .get("Laser", {})
        )
        return LaserSystemState(
            lasers=[LaserState(**laser_data) for laser_data in laser_array_data]
        )

    def _parse_all_channels(self) -> list[ChannelMetadata]:
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
            )
            for i, lif_channel in enumerate(self.image_description.lif_channels)
        ]

    def _parse_channel_metadata(
        self,
        lif_channel: _LifChannel,
        channel: Channel | None = None,
    ) -> ChannelMetadata:
        """Parse metadata for a specific channel."""
        if channel is None:
            channel = self._infer_channel(lif_channel)

        resolution = self._parse_nominal_dimensions()
        measured = self._parse_measured_dimensions()
        acquisition = self._parse_acquisition_settings()
        optics = self._parse_microscope_settings()

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
            LightSourceType.DIODE,
            LightSourceType.WLL,
        ):
            active_laser_state = self.laser_system_state.get_laser_by_type(active_lasers[0])
            return self._infer_channel_from_laser_state(active_laser_state)

        return self._infer_channel_from_detector(lif_channel, active_lasers)

    def _infer_channel_from_laser_state(self, laser_state: LaserState) -> Channel:
        """Infer channel from laser state using excitation wavelength."""
        if laser_state.LightSourceType == LightSourceType.CRS:
            raise ValueError("Cannot infer channel from CRS laser")

        # Can reasonably infer channel only in the case where either the UV or WLL laser is ON
        excitation_wavelength_nm = self._extract_wavelength_value(laser_state.WavelengthDouble)
        try:
            return Channel.from_excitation_wavelength(
                excitation_wavelength_nm, name=laser_state.LightSourceType.name
            )
        except ValueError:
            warnings.warn(
                f"Parsed excitation wavelength {excitation_wavelength_nm} nm outside accepted "
                "range for Channel inference. Pass a Channel instance to prevent this warning.",
                stacklevel=2,
            )
            return Channel(name=laser_state.LightSourceType.name)

    def _infer_channel_from_detector(
        self,
        lif_channel: _LifChannel,
        active_lasers: list[LightSourceType],
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
                LightSourceType.WLL
                if LightSourceType.WLL in active_lasers
                else LightSourceType.DIODE
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
            laser_state = self.laser_system_state.get_laser_by_type(LightSourceType.CRS)
            pump_wavelength_nm = self._extract_wavelength_value(laser_state.WavelengthDouble)

            if channel == CARS:
                # CARS detects anti-Stokes wavelength
                stokes_wavelength_nm = 1031.7  # Fixed Stokes wavelength for CRS laser
                emission_nm = float(
                    calculate_antistokes_wavelength(pump_wavelength_nm, stokes_wavelength_nm)
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
        dimensions = DimensionFlags(0)

        if "T" in self.sizes and self.sizes["T"] > 1:
            dimensions |= DimensionFlags.TIMELAPSE
        if "Z" in self.sizes and self.sizes["Z"] > 1:
            dimensions |= DimensionFlags.Z_STACK
        if "S" in self.sizes and self.sizes["S"] > 1:
            dimensions |= DimensionFlags.RGB
        if "λ" in self.sizes and self.sizes["λ"] > 1:
            dimensions |= DimensionFlags.SPECTRAL
        if "Λ" in self.sizes and self.sizes["Λ"] > 1:
            dimensions |= DimensionFlags.SPECTRAL
        if "M" in self.sizes and self.sizes["M"] > 1:
            dimensions |= DimensionFlags.MONTAGE

        return dimensions

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
        xy_step_um = 1e6 * (x_dim.step + y_dim.step) / 2

        # Optional Z dimension
        z_size_px, z_step_um = None, None
        if self.dimensions.is_zstack:
            z_dim = self._find_dimension(3)
            z_size_px = z_dim.number_of_elements
            z_step_um = 1e6 * z_dim.step

        # Optional T dimension
        t_size_px, t_step_ms = None, None
        if self.dimensions.is_timelapse:
            t_dim = self._find_dimension(4)
            t_size_px = t_dim.number_of_elements
            t_step_ms = 1e3 * t_dim.step

        # Optional wavelength dimension (try Λ then λ)
        w_size_px, w_step_nm = None, None
        if self.dimensions.is_spectral:
            for dim_id, size_key in [(9, "Λ"), (5, "λ")]:
                if size_key in self.sizes and self.sizes[size_key] > 1:
                    w_dim = self._find_dimension(dim_id)
                    w_size_px = w_dim.number_of_elements
                    w_step_nm = 1e9 * w_dim.step
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
            z_values_um = self.image.coords["Z"]

        t_values_ms = None
        if self.dimensions.is_timelapse:
            t_values_ms = self.image.coords["T"]

        w_values_nm = None
        if self.dimensions.is_spectral:
            laser_values_data = (
                self.image.attrs.get("LaserValues", {})
                .get("Laser", {})
                .get("StagePosition", {})
                .get("LaserValues", {})
            )
            laser_values = LaserValueCollection([LaserValue(**item) for item in laser_values_data])
            w_values_nm = laser_values.wavelengths_nm

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

        # Calculate exposure time from per-pixel dwell time and spatial dimensions
        # Total time in seconds: pixel_dwell_time_s * X * Y, then convert to milliseconds
        exposure_time_ms = pixel_dwell_time_s * self.sizes["X"] * self.sizes["Y"] * 1e3

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
            # Convert to nm if it looks like it's in meters (< 1)
            return wavelength * 1e9 if wavelength < 1 else wavelength
        except (ValueError, TypeError) as ex:
            raise ValueError(f"Cannot determine wavelength from {value}") from ex
