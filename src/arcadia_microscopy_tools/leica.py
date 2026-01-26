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
    """List all image names contained in a LIF file.

    Args:
        lif_path: Path to the Leica LIF file

    Returns:
        List of image names in the file
    """
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

    # Map of (detector_name, beam_route) to Channel for automatic channel detection
    _CHANNEL_DETECTION_MAP = {
        ("F-SRS", None): SRS,  # beam_route typically "10;0" but not checked
        ("HyD NDD 1", "20;21"): CARS,
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
        """Parse the ImageDescription XML element into structured data.

        Args:
            image_description_element: The <ImageDescription> XML element

        Returns:
            ImageDescription containing channels and dimensions
        """
        channels_element = image_description_element.find("Channels")
        dimensions_element = image_description_element.find("Dimensions")
        if channels_element is None or dimensions_element is None:
            raise ValueError("Expected <Channels> and <Dimensions> under <ImageDescription>")

        lif_channels = self._parse_channels_from_xml(channels_element)
        lif_dimensions = self._parse_dimensions_from_xml(dimensions_element)

        return ImageDescription(lif_channels, lif_dimensions)

    def _parse_channels_from_xml(self, channels_element: ET.Element) -> list[LifChannel]:
        """Parse channel descriptions from the <Channels> XML element.

        Args:
            channels_element: The <Channels> XML element

        Returns:
            List of LifChannel objects
        """
        lif_channels: list[LifChannel] = []
        for channel_element in channels_element.findall("ChannelDescription"):
            # Extract channel properties
            props: dict[str, str] = {}
            for prop in channel_element.findall("ChannelProperty"):
                key_element = prop.find("Key")
                value_element = prop.find("Value")
                if key_element is None or value_element is None or key_element.text is None:
                    continue
                props[key_element.text] = value_element.text or ""

            # Build LifChannel from XML attributes
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

        return lif_channels

    def _parse_dimensions_from_xml(self, dimensions_element: ET.Element) -> list[LifDimension]:
        """Parse dimension descriptions from the <Dimensions> XML element.

        Args:
            dimensions_element: The <Dimensions> XML element

        Returns:
            List of LifDimension objects
        """
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

        return lif_dimensions

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
        if channel is None:
            channel = self._infer_channel(lif_channel)

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

    def _infer_channel(self, lif_channel: LifChannel) -> Channel:
        """Infer channel type from detector metadata.

        Args:
            lif_channel: The LIF channel to infer from

        Returns:
            The inferred Channel

        Raises:
            ValueError: If channel cannot be determined from metadata
        """
        detector_name = lif_channel.properties.get("DetectorName")
        beam_route = lif_channel.properties.get("BeamRoute")

        # Try exact match with beam route
        if (detector_name, beam_route) in self._CHANNEL_DETECTION_MAP:
            return self._CHANNEL_DETECTION_MAP[(detector_name, beam_route)]

        # Try match without beam route
        if (detector_name, None) in self._CHANNEL_DETECTION_MAP:
            return self._CHANNEL_DETECTION_MAP[(detector_name, None)]

        raise ValueError(
            f"Could not determine channel from DetectorName: {detector_name}, "
            f"BeamRoute: {beam_route}. Please provide channels list explicitly."
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
        # Find and validate X and Y dimensions
        lif_dimension_x = self._find_dimension(1)
        lif_dimension_y = self._find_dimension(2)
        self._validate_dimension(lif_dimension_x, "X", self.sizes["X"])
        self._validate_dimension(lif_dimension_y, "Y", self.sizes["Y"])

        # Calculate pixel size using step property and convert to microns
        pixel_size_um_x = 1e6 * lif_dimension_x.step
        pixel_size_um_y = 1e6 * lif_dimension_y.step
        pixel_size_um = (pixel_size_um_x + pixel_size_um_y) / 2

        # Calculate thickness and step size for z
        thickness_px = None
        z_step_size_um = None
        if self.dimensions.is_zstack:
            lif_dimension_z = self._find_dimension(3)
            self._validate_dimension(lif_dimension_z, "Z", self.sizes["Z"])
            thickness_px = lif_dimension_z.number_of_elements
            z_step_size_um = 1e6 * lif_dimension_z.step

        return PhysicalDimensions(
            height_px=self.sizes["Y"],
            width_px=self.sizes["X"],
            pixel_size_um=pixel_size_um,
            thickness_px=thickness_px,
            z_step_size_um=z_step_size_um,
        )

    def _find_dimension(self, dim_id: int) -> LifDimension:
        """Find a dimension by its ID.

        Args:
            dim_id: The dimension ID to find

        Returns:
            The LifDimension with the given ID

        Raises:
            ValueError: If dimension is not found
        """
        dimension = next(
            (d for d in self.image_description.lif_dimensions if d.dim_id == dim_id), None
        )
        if dimension is None:
            raise ValueError(f"Missing dimension (dim_id={dim_id}) in LIF metadata")
        return dimension

    def _validate_dimension(
        self, dimension: LifDimension, dim_name: str, expected_size: int
    ) -> None:
        """Validate a dimension's unit and size.

        Args:
            dimension: The dimension to validate
            dim_name: Human-readable name for error messages (e.g., "X", "Y", "Z")
            expected_size: Expected number of elements

        Raises:
            ValueError: If validation fails
        """
        if dimension.unit != "m":
            raise ValueError(f"Expected {dim_name} dimension unit 'm' but got: {dimension.unit}")

        if dimension.number_of_elements != expected_size:
            raise ValueError(
                f"{dim_name} dimension mismatch: lif_dimension has {dimension.number_of_elements} "
                f"but sizes has {expected_size}"
            )

    def _parse_acquisition_settings(self, channel: Channel) -> AcquisitionSettings:
        """Parse acquisition settings from LIF metadata."""
        wavelengths_nm = None
        if self.dimensions.is_spectral or channel in (CARS, SRS):
            extractor = _WavelengthExtractor(self.image, self.sizes)
            wavelengths_nm = extractor.extract_wavelengths(channel)

        return AcquisitionSettings(
            exposure_time_ms=-1,
            zoom=None,
            binning=None,
            frame_intervals_ms=None,
            wavelengths_nm=wavelengths_nm,
        )

    def _parse_microscope_settings(self) -> MicroscopeSettings:
        """Parse microscope settings from LIF metadata."""
        microscope_data = self.image.attrs.get("HardwareSetting", {}).get(
            "ATLConfocalSettingDefinition", {}
        )

        magnification = int(microscope_data.get("Magnification", -1))
        numerical_aperture = float(microscope_data.get("NumericalAperture", -1.0))
        objective = microscope_data.get("ObjectiveName")

        return MicroscopeSettings(
            magnification=magnification,
            numerical_aperture=numerical_aperture,
            objective=objective,
            light_source=None,
            power_mw=None,
        )

    @staticmethod
    def _required(e: ET.Element, name: str) -> str:
        v = e.get(name)
        if v is None:
            raise ValueError(f"Missing attribute {name!r} on <{e.tag}>")
        return v


class _WavelengthExtractor:
    """Extracts wavelength information from LIF file metadata.

    Handles the complexity of finding wavelength data in various locations
    within the LIF metadata structure, which varies by acquisition mode.
    """

    def __init__(self, image: liffile.LifImageABC, sizes: dict[str, int]):
        self.image = image
        self.sizes = sizes

    def extract_wavelengths(self, channel: Channel) -> FloatArray | None:
        """Find laser wavelength information for the given channel.

        Tries multiple strategies depending on the acquisition mode:
            1. For Lambda (Λ) scans: checks LaserValues path
            2. For CARS/SRS: checks HardwareSetting LaserArray
            3. Falls back to recursive search through all attrs

        Args:
            channel: The channel being parsed (determines which paths to check)

        Returns:
            Array of wavelengths in nanometers, or None if no wavelengths found.
        """
        # Strategy 1: Lambda (Λ) scans - check LaserValues path
        if "Λ" in self.sizes and self.sizes["Λ"] > 1:
            wavelengths = self._try_lambda_scan_path()
            if wavelengths is not None:
                return wavelengths

        # Strategy 2: CARS/SRS channels - check HardwareSetting path
        if channel in (CARS, SRS):
            wavelengths = self._try_cars_srs_path()
            if wavelengths is not None:
                return wavelengths

        # Strategy 3: Recursive search through all attrs
        return self._recursive_search()

    def _try_lambda_scan_path(self) -> FloatArray | None:
        """Try to extract wavelengths from Lambda scan specific path."""
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
            pass
        return None

    def _try_cars_srs_path(self) -> FloatArray | None:
        """Try to extract wavelengths from CARS/SRS specific path."""
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
            pass
        return None

    def _recursive_search(self) -> FloatArray | None:
        """Fall back to searching through all metadata recursively."""
        results = []
        self._search_wavelengths(self.image.attrs, results)

        if results:
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
