from __future__ import annotations
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import liffile
import numpy as np

from .channels import CARS, SRS, Channel
from .metadata_structures import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MicroscopeSettings,
    PhysicalDimensions,
)
from .microscopy import ImageMetadata
from .typing import FloatArray


def _as_int(s: str, *, ctx: str = "") -> int:
    try:
        return int(s)
    except ValueError as ex:
        raise ValueError(f"Expected int{(' for ' + ctx) if ctx else ''}, got {s!r}") from ex


def _as_float(s: str, *, ctx: str = "") -> float:
    try:
        return float(s)
    except ValueError as ex:
        raise ValueError(f"Expected float{(' for ' + ctx) if ctx else ''}, got {s!r}") from ex


@dataclass(frozen=True)
class LifChannel:
    """Recreated from liffile where it is not exposed and ignores properties"""

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
    properties: Mapping[str, str]


@dataclass(frozen=True)
class LifDimension:
    """Recreated from liffile where it is not exposed"""

    dim_id: int
    number_of_elements: int
    origin: float
    length: float
    unit: str
    bit_inc: int
    bytes_inc: int

    @property
    def step(self) -> float:
        return self.length / self.number_of_elements


@dataclass(frozen=True)
class ImageDescription:
    lif_channels: list[LifChannel]
    lif_dimensions: list[LifDimension]


def list_image_names(lif_path: Path) -> list[str]:
    """"""
    with liffile.LifFile(lif_path) as f:
        image_names = [image.name for image in f.images]
    return image_names


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


class _LeicaMetadataParser:
    """Parser for extracting metadata from Leica LIF files."""

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
                    f"Missing attrs metadata for image {self.lif_path}/{self.image_name}"
                )

            self.sizes = self.image.sizes
            self.dimensions = self._get_dimension_flags()
            self.timestamp = self._parse_timestamp()

            # Parse image description
            image_description_element = self.image.xml_element.find("./Data/Image/ImageDescription")
            if image_description_element is None:
                raise ValueError(
                    f"Missing image description metadata {self.lif_path}/{self.image_name}"
                )
            self.image_description = self.parse_image_description(image_description_element)

            channel_metadata_list = self._parse_all_channels()
            return ImageMetadata(self.sizes, channel_metadata_list)

    def parse_image_description(self, image_description_element: ET.Element) -> ImageDescription:
        channels_element = image_description_element.find("Channels")
        dimensions_element = image_description_element.find("Dimensions")
        if channels_element is None or dimensions_element is None:
            raise ValueError("Expected <Channels> and <Dimensions> under <ImageDescription>")

        lif_channels: list[LifChannel] = []
        for channel_element in channels_element.findall("ChannelDescription"):
            props: dict[str, str] = {}
            for prop in channel_element.findall("ChannelProperty"):
                key_element = prop.find("Key")
                value_element = prop.find("Value")
                if key_element is None or value_element is None or key_element.text is None:
                    continue
                props[key_element.text] = value_element.text or ""

            lif_channels.append(
                LifChannel(
                    data_type=_as_int(self._required(channel_element, "DataType"), ctx="DataType"),
                    channel_tag=_as_int(
                        self._required(channel_element, "ChannelTag"), ctx="ChannelTag"
                    ),
                    resolution=_as_int(
                        self._required(channel_element, "Resolution"), ctx="Resolution"
                    ),
                    lut_name=self._required(channel_element, "LUTName"),
                    bytes_inc=_as_int(self._required(channel_element, "BytesInc"), ctx="BytesInc"),
                    bit_inc=_as_int(self._required(channel_element, "BitInc"), ctx="BitInc"),
                    min_value=_as_float(self._required(channel_element, "Min"), ctx="Min"),
                    max_value=_as_float(self._required(channel_element, "Max"), ctx="Max"),
                    unit=channel_element.get("Unit", ""),
                    name_of_measured_quantity=channel_element.get("NameOfMeasuredQuantity", ""),
                    properties=props,
                )
            )

        lif_dimensions: list[LifDimension] = []
        for dimension_element in dimensions_element.findall("DimensionDescription"):
            lif_dimensions.append(
                LifDimension(
                    dim_id=_as_int(self._required(dimension_element, "DimID"), ctx="DimID"),
                    number_of_elements=_as_int(
                        self._required(dimension_element, "NumberOfElements"),
                        ctx="NumberOfElements",
                    ),
                    origin=_as_float(self._required(dimension_element, "Origin"), ctx="Origin"),
                    length=_as_float(self._required(dimension_element, "Length"), ctx="Length"),
                    unit=self._required(dimension_element, "Unit"),
                    bit_inc=_as_int(self._required(dimension_element, "BitInc"), ctx="BitInc"),
                    bytes_inc=_as_int(
                        self._required(dimension_element, "BytesInc"), ctx="BytesInc"
                    ),
                )
            )

        return ImageDescription(lif_channels, lif_dimensions)

    def _parse_all_channels(self) -> list[ChannelMetadata]:
        """Parse metadata for all channels in the LIF image."""
        # Validate channels list length if provided
        num_channels = len(self.image_description.lif_channels)
        if self.channels is not None and len(self.channels) != num_channels:
            raise ValueError(
                f"Expected {num_channels} channels but got {len(self.channels)} in channels list"
            )

        channel_metadata_list = []
        for i, lif_channel in enumerate(self.image_description.lif_channels):
            channel = self.channels[i] if self.channels else None
            channel_metadata = self._parse_channel_metadata(lif_channel, channel)
            channel_metadata_list.append(channel_metadata)

        return channel_metadata_list

    def _parse_channel_metadata(
        self,
        lif_channel: LifChannel,
        channel: Channel | None = None,
    ) -> ChannelMetadata:
        """Parse metadata for a specific channel."""
        # Try to resolve channel
        if channel is None:
            detector_name = lif_channel.properties.get("DetectorName")
            beam_route = lif_channel.properties.get("BeamRoute")
            if detector_name == "F-SRS":  # beam_route == "10;0"
                channel = SRS
            elif detector_name == "HyD NDD 1" and beam_route == "20;21":
                channel = CARS
            else:
                raise ValueError(
                    f"Could not determine channel from DetectorName: {detector_name}. "
                    f"Please provide channels list explicitly."
                )

        resolution = self._parse_physical_dimensions()
        acquisition = self._parse_acquisition_settings(channel)
        optics = self._parse_microscope_settings()

        return ChannelMetadata(
            channel=channel,
            timestamp=self.timestamp,
            dimensions=self.dimensions,
            resolution=resolution,
            acquisition=acquisition,
            optics=optics,
        )

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

        if "Λ" in self.sizes and self.sizes["Λ"] > 1:
            dimensions |= DimensionFlags.SPECTRAL

        if "M" in self.sizes and self.sizes["M"] > 1:
            dimensions |= DimensionFlags.MONTAGE

        return dimensions

    def _parse_timestamp(self) -> datetime:
        """Parse timestamp from LIF metadata."""
        return self._lif.images[self.image_name].timestamps[0]

    def _parse_physical_dimensions(self) -> PhysicalDimensions:
        """Parse physical dimensions from LIF metadata.

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
        # Find X and Y dimensions
        lif_dimension_x = next(
            (d for d in self.image_description.lif_dimensions if d.dim_id == 1), None
        )
        lif_dimension_y = next(
            (d for d in self.image_description.lif_dimensions if d.dim_id == 2), None
        )

        if lif_dimension_x is None:
            raise ValueError("Missing X dimension (dim_id=1) in LIF metadata")
        if lif_dimension_y is None:
            raise ValueError("Missing Y dimension (dim_id=2) in LIF metadata")

        # Validate dimensions match sizes
        if lif_dimension_x.number_of_elements != self.sizes["X"]:
            raise ValueError(
                f"X dimension mismatch: lif_dimension has {lif_dimension_x.number_of_elements} "
                f"but sizes has {self.sizes['X']}"
            )
        if lif_dimension_y.number_of_elements != self.sizes["Y"]:
            raise ValueError(
                f"Y dimension mismatch: lif_dimension has {lif_dimension_y.number_of_elements} "
                f"but sizes has {self.sizes['Y']}"
            )

        # Check that units are in meters
        if lif_dimension_x.unit != "m":
            raise ValueError(f"Expected X dimension unit 'm' but got: {lif_dimension_x.unit}")
        if lif_dimension_y.unit != "m":
            raise ValueError(f"Expected Y dimension unit 'm' but got: {lif_dimension_y.unit}")

        # Calculate pixel size using step property and convert to microns
        pixel_size_um_x = 1e6 * lif_dimension_x.step
        pixel_size_um_y = 1e6 * lif_dimension_y.step
        pixel_size_um = (pixel_size_um_x + pixel_size_um_y) / 2

        # Calculate thickness and step size for z
        thickness_px = None
        z_step_size_um = None
        if self.dimensions.is_zstack:
            lif_dimension_z = next(
                (d for d in self.image_description.lif_dimensions if d.dim_id == 3), None
            )
            if lif_dimension_z is None:
                raise ValueError("Missing Z dimension (dim_id=3) in LIF metadata for z-stack")

            # Validate Z dimension
            if lif_dimension_z.unit != "m":
                raise ValueError(f"Expected Z dimension unit 'm' but got: {lif_dimension_z.unit}")
            if lif_dimension_z.number_of_elements != self.sizes["Z"]:
                raise ValueError(
                    f"Z dimension mismatch: lif_dimension has {lif_dimension_z.number_of_elements} "
                    f"but sizes has {self.sizes['Z']}"
                )

            thickness_px = lif_dimension_z.number_of_elements
            z_step_size_um = 1e6 * lif_dimension_z.step

        return PhysicalDimensions(
            height_px=self.sizes["Y"],
            width_px=self.sizes["X"],
            pixel_size_um=pixel_size_um,
            thickness_px=thickness_px,
            z_step_size_um=z_step_size_um,
        )

    def _parse_acquisition_settings(self, channel: Channel) -> AcquisitionSettings:
        """Parse acquisition settings from LIF metadata."""
        wavelengths_nm = None
        if self.dimensions.is_spectral or channel in (CARS, SRS):
            wavelengths_nm = self._find_laser_wavelengths(channel)

        return AcquisitionSettings(
            exposure_time_ms=-1,
            zoom=None,
            binning=None,
            frame_intervals_ms=None,
            wavelengths_nm=wavelengths_nm,
        )

    def _parse_microscope_settings(self) -> MicroscopeSettings:
        """Parse microscope settings from LIF metadata."""
        magnification = -1
        numerical_aperture = -1.0
        objective = None

        try:
            microscope_data = self.image.attrs["HardwareSetting"]["ATLConfocalSettingDefinition"]

            # Try to extract magnification
            if "Magnification" in microscope_data:
                try:
                    magnification = int(microscope_data["Magnification"])
                except (ValueError, TypeError):
                    pass

            # Try to extract numerical aperture
            if "NumericalAperture" in microscope_data:
                try:
                    numerical_aperture = float(microscope_data["NumericalAperture"])
                except (ValueError, TypeError):
                    pass

            # Try to extract objective name
            if "ObjectiveName" in microscope_data:
                objective_name = microscope_data["ObjectiveName"]
                if objective_name is not None and isinstance(objective_name, str):
                    objective = objective_name.strip()

        except (KeyError, TypeError):
            # HardwareSetting path doesn't exist, use defaults
            pass

        return MicroscopeSettings(
            magnification=magnification,
            numerical_aperture=numerical_aperture,
            objective=objective,
            light_source=None,
            power_mw=None,
        )

    def _find_laser_wavelengths(self, channel: Channel) -> FloatArray | None:
        """Recursively search for laser wavelength information in metadata.

        This searches through the attrs dictionary structure to find wavelength information
        which can be located in various places depending on acquisition mode.

        Args:
            channel: The channel being parsed (used to determine which paths to check)

        Returns:
            Array of wavelengths in nanometers, or None if no wavelengths found.
        """
        # For Lambda (Λ) scans, try the specific LaserValues path first
        if "Λ" in self.sizes and self.sizes["Λ"] > 1:
            try:
                laser_values = self.image.attrs["LaserValues"]["Laser"]["StagePosition"][
                    "LaserValues"
                ]
                if laser_values is not None:
                    results = []
                    self._search_wavelengths(laser_values, results)
                    if results:
                        wavelengths = sorted(set(results))
                        return np.array(wavelengths, dtype=np.float64)
            except (KeyError, TypeError):
                # Path doesn't exist, fall through to next strategy
                pass

        # For CARS/SRS channels, check HardwareSetting path
        if channel in (CARS, SRS):
            try:
                laser_array = self.image.attrs["HardwareSetting"]["ATLConfocalSettingDefinition"][
                    "LaserArray"
                ]["Laser"]
                # Try to find the CRS laser (often at index 2)
                if isinstance(laser_array, list) and len(laser_array) > 2:
                    crs_laser = laser_array[2]
                    if crs_laser.get("LaserName") == "CRS":
                        results = []
                        self._search_wavelengths(crs_laser, results)
                        if results:
                            wavelengths = sorted(set(results))
                            return np.array(wavelengths, dtype=np.float64)
            except (KeyError, TypeError, IndexError):
                # Path doesn't exist, fall through to recursive search
                pass

        # Fall back to recursive search through all attrs
        results = []
        self._search_wavelengths(self.image.attrs, results)

        if results:
            # Flatten and deduplicate wavelengths, then sort
            wavelengths = sorted(set(results))
            return np.array(wavelengths, dtype=np.float64)

        return None

    def _search_wavelengths(
        self, data: dict | list | str | int | float, results: list[float]
    ) -> None:
        """Recursively search through metadata structure for wavelength values.

        Args:
            data: Current node in the metadata structure
            results: List to accumulate found wavelengths (modified in place)
        """
        if isinstance(data, dict):
            # Look for wavelength fields in this dict
            # Prefer WavelengthDouble over Wavelength if both exist
            for key in ["WavelengthDouble", "WavelengthBegin", "WavelengthEnd", "Excitation"]:
                if key in data:
                    wavelength = self._extract_wavelength_value(data[key])
                    if wavelength is not None:
                        results.append(wavelength)

            # Only check "Wavelength" if "WavelengthDouble" wasn't found
            if "Wavelength" in data and "WavelengthDouble" not in data:
                wavelength = self._extract_wavelength_value(data["Wavelength"])
                if wavelength is not None:
                    results.append(wavelength)

            # If we have begin and end, generate a range
            if "WavelengthBegin" in data and "WavelengthEnd" in data:
                begin = self._extract_wavelength_value(data["WavelengthBegin"])
                end = self._extract_wavelength_value(data["WavelengthEnd"])
                if begin is not None and end is not None and "WavelengthStep" in data:
                    step = self._extract_wavelength_value(data["WavelengthStep"])
                    if step is not None and step > 0:
                        # Generate wavelength range
                        current = begin
                        while current <= end:
                            results.append(current)
                            current += step
                        return  # Don't recurse further into this branch

            # Recurse into child values
            for value in data.values():
                self._search_wavelengths(value, results)

        elif isinstance(data, list):
            # Recurse into list items
            for item in data:
                self._search_wavelengths(item, results)

    def _extract_wavelength_value(self, value: str | int | float | dict | list) -> float | None:
        """Extract a wavelength value and convert to nanometers if needed.

        Args:
            value: The value which might contain wavelength information

        Returns:
            Wavelength in nanometers, or None if value cannot be parsed
        """
        # If it's already a number, assume it's in appropriate units
        if isinstance(value, int | float):
            wavelength = float(value)
            # Convert to nm if it looks like it's in meters (< 1)
            if wavelength < 1:
                wavelength *= 1e9
            return wavelength

        # If it's a string, try to parse it
        if isinstance(value, str):
            try:
                wavelength = float(value)
                # Convert to nm if it looks like it's in meters (< 1)
                if wavelength < 1:
                    wavelength *= 1e9
                return wavelength
            except (ValueError, TypeError):
                pass

        return None

    @staticmethod
    def _required(e: ET.Element, name: str) -> str:
        v = e.get(name)
        if v is None:
            raise ValueError(f"Missing attribute {name!r} on <{e.tag}>")
        return v
