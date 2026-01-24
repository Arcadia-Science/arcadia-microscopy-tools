from __future__ import annotations
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import liffile

from .channels import CARS, SRS, Channel
from .metadata_structures import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MicroscopeSettings,
    PhysicalDimensions,
)
from .microscopy import ImageMetadata


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
        acquisition = self._parse_acquisition_settings()
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
            X: dim_id = 1
            Y: dim_id = 2
            Z: dim_id = 3
            ...
            Λ: dim_id = 9
        """
        lif_dimension_x = next(d for d in self.image_description.lif_dimensions if d.dim_id == 1)
        lif_dimension_y = next(d for d in self.image_description.lif_dimensions if d.dim_id == 2)

        # Check that units are in meters
        units = [d.unit for d in [lif_dimension_x, lif_dimension_y]]
        if not all(unit == "m" for unit in units):
            raise ValueError(f"Expected lengths in 'm' for physical dimensions but got: {units}")

        # Calculate pixel size and convert to microns
        pixel_size_um_x = 1e6 * lif_dimension_x.length / lif_dimension_x.number_of_elements
        pixel_size_um_y = 1e6 * lif_dimension_y.length / lif_dimension_y.number_of_elements
        pixel_size_um = (pixel_size_um_x + pixel_size_um_y) / 2

        # Calculate thickness and step size for z
        thickness_px = None
        z_step_size_um = None
        if self.dimensions.is_zstack:
            lif_dimension_z = next(
                d for d in self.image_description.lif_dimensions if d.dim_id == 3
            )
            # Check that unit is in meters
            if lif_dimension_z.unit != "m":
                raise ValueError(f"Expected length in 'm' but got: {lif_dimension_z.unit}")

            thickness_px = lif_dimension_z.number_of_elements
            z_step_size_um = 1e6 * lif_dimension_z.length / thickness_px

        return PhysicalDimensions(
            height_px=self.sizes["Y"],
            width_px=self.sizes["X"],
            pixel_size_um=pixel_size_um,
            thickness_px=thickness_px,
            z_step_size_um=z_step_size_um,
        )

    def _parse_acquisition_settings(self) -> AcquisitionSettings:
        """Parse acquisition settings from LIF metadata."""
        import numpy as np

        wavelengths_nm = np.array([-1.0])

        return AcquisitionSettings(
            exposure_time_ms=-1,
            zoom=None,
            binning=None,
            frame_intervals_ms=None,
            wavelengths_nm=wavelengths_nm if self.dimensions.is_spectral else None,
        )

    def _parse_microscope_settings(self) -> MicroscopeSettings:
        """Parse microscope settings from LIF metadata."""
        return MicroscopeSettings(
            magnification=-1,
            numerical_aperture=-1,
            objective=None,
            light_source=None,
            power_mw=None,
        )

    @staticmethod
    def _required(e: ET.Element, name: str) -> str:
        v = e.get(name)
        if v is None:
            raise ValueError(f"Missing attribute {name!r} on <{e.tag}>")
        return v
