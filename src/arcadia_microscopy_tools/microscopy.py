from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import nd2
from nd2.structures import Color

from .typing import UInt16Array

CHANNEL_EXCITATION_PEAKS_NM = {
    "DAPI": 405,
    "FITC": 488,
    "TRITC": 561,
}
CHANNEL_EMISSION_PEAKS_NM = {
    "DAPI": 450,
    "FITC": 512,
    "TRITC": 595,
}
CHANNEL_COLORS_RGB = {
    "BF": Color(255, 255, 255),
    "DAPI": Color(0, 51, 255),
    "FITC": Color(7, 255, 0),
    "TRITC": Color(255, 191, 0),
}

NIKON_OPTICAL_CONFIGS_MAP = {
    "Mono": "BF",
    "DAPI": "DAPI",
    "FITC BP": "FITC",
    "TRITC BP": "TRITC",
    "GFP 488 nm": "FITC",
}


class Channel(Enum):
    """Common microscopy channels.

    Represents standard fluorescence microscopy channels and brightfield imaging modes.
    Each channel has associated excitation/emission wavelengths and display colors when applicable.
    """

    BF = auto()  # brightfield
    DIC = auto()  # differential interference contrast
    DAPI = auto()  # blue
    FITC = auto()  # green
    TRITC = auto()  # red
    CY5 = auto()  # far red

    def __init__(self, *args):
        super().__init__()
        self.excitation_nm = CHANNEL_EXCITATION_PEAKS_NM.get(self.name)
        self.emission_nm = CHANNEL_EMISSION_PEAKS_NM.get(self.name)
        self.color = CHANNEL_COLORS_RGB.get(self.name)

    @classmethod
    def from_optical_config_name(cls, optical_config: str) -> Channel:
        """Get the Channel enum from the optical configuration name.

        Args:
            optical_config: The name of the optical configuration.

        Returns:
            Channel: The corresponding Channel enum.

        Raises:
            ValueError: If the optical configuration is not recognized.
        """
        if optical_config in NIKON_OPTICAL_CONFIGS_MAP:
            channel = NIKON_OPTICAL_CONFIGS_MAP[optical_config]
            return cls[channel]
        else:
            raise ValueError(f"{optical_config} is not a known optical configuration.")


@dataclass
class ChannelMetadata:
    """Metadata for a single channel of microscopy image.

    Contains physical and acquisition parameters for a specific imaging channel.
    """

    channel: Channel | None = None
    timestamp: datetime | None = None
    height_px: int | None = None
    width_px: int | None = None
    thickness_px: int | None = None
    pixel_size_um: tuple[float, float] | None = None
    z_step_size_um: float | None = None
    magnification: float | None = None
    numerical_aperture: float | None = None
    zoom: float | None = None
    binning: str | None = None
    exposure_time_ms: float | None = None
    laser_power_pct: float | None = None

    @classmethod
    def from_nd2_path(
        cls,
        nd2_path: Path | str,
        channel_index: int = 0,
    ) -> ChannelMetadata:
        """Extract metadata for a specific channel from a Nikon ND2 file.

        Args:
            nd2_path: Path to the Nikon ND2 file.
            channel_index: Index of the channel to extract metadata for (default: 0).

        Returns:
            ChannelMetadata: Metadata for the specified channel.

        The strategy is to first check `nd2.ND2File.metadata` for relevant metadata fields as this
        appears to be the most reliable and most straightforward source to extract. Some relevant
        metadata fields such as the timestamp, binning, power, and exposure time are nowhere to be
        found in `nd2.ND2File.metadata`, so for these fields we parse `nd2.ND2File.text_info`
        instead.
        """
        # Extract metadata using nd2
        with nd2.ND2File(nd2_path) as nd2f:
            nd2_channel = nd2f.metadata.channels[channel_index]
            text_info = nd2f.text_info

        # Get channel from Nikon optical configuration
        optical_config = nd2_channel.channel.name
        channel = Channel.from_optical_config_name(optical_config)

        # Certain metadata fields can only be found in certain sections of `text_info`
        sample_text = _extract_sample_from_text_info(text_info, sample_index=channel_index + 1)
        plane_text = _extract_plane_from_text_info(text_info, plane_index=channel_index + 1)

        # Parse these specific metadata fields from the section of `text_info`
        timestamp = _parse_timestamp_from_text_info(text_info)
        binning = _parse_binning_from_sample(sample_text)
        exposure_time_ms = _parse_exposure_time_from_sample(sample_text)
        laser_power_pct = _parse_laser_power_from_plane(plane_text)

        return cls(
            channel=channel,
            timestamp=timestamp,
            height_px=nd2_channel.volume.voxelCount[0],
            width_px=nd2_channel.volume.voxelCount[1],
            thickness_px=nd2_channel.volume.voxelCount[2],
            pixel_size_um=nd2_channel.volume.axesCalibration[:2],
            z_step_size_um=nd2_channel.volume.axesCalibration[2],
            magnification=nd2_channel.microscope.objectiveMagnification,
            numerical_aperture=nd2_channel.microscope.objectiveNumericalAperture,
            zoom=nd2_channel.microscope.zoomMagnification,
            binning=binning,
            exposure_time_ms=exposure_time_ms,
            laser_power_pct=laser_power_pct,
        )


@dataclass
class ImageMetadata:
    """Metadata for a complete microscopy image.

    Contains metadata for all channels in the image.
    """

    channels: list[ChannelMetadata]

    @classmethod
    def from_nd2_path(
        cls,
        nd2_path: Path | str,
    ) -> ImageMetadata:
        """Create ImageMetadata from a Nikon ND2 file.

        Args:
            nd2_path: Path to the Nikon ND2 file.

        Returns:
            ImageMetadata: Metadata for all channels in the image.
        """
        with nd2.ND2File(nd2_path) as nd2f:
            num_channels = nd2f.metadata.contents.channelCount
        channel_metadatas = []
        for i in range(num_channels):
            channel_metadata = ChannelMetadata.from_nd2_path(
                nd2_path,
                channel_index=i,
            )
            channel_metadatas.append(channel_metadata)
        return cls(channel_metadatas)


@dataclass
class Metadata:
    """Combined metadata for a microscopy image of a sample.

    Contains both sample-specific metadata and image acquisition metadata.
    """

    sample: dict[str, Any] | None = None
    image: ImageMetadata | None = None


@dataclass
class MicroscopyImage:
    """Dataclass for microscopy image data.

    Contains both the image intensity data and associated metadata for all channels.
    Provides methods to access specific channel data.
    """

    intensities: UInt16Array
    metadata: Metadata

    @classmethod
    def from_nd2_path(
        cls,
        nd2_path: Path | str,
        sample_metadata: dict[str, Any] | None = None,
    ) -> MicroscopyImage:
        """Create MicroscopyImage from a Nikon ND2 file.

        Args:
            nd2_path: Path to the Nikon ND2 file.
            sample_metadata: Optional dictionary containing sample-specific metadata.

        Returns:
            MicroscopyImage: A new microscopy image with intensity data and metadata.
        """
        intensities = nd2.imread(nd2_path)
        image_metadata = ImageMetadata.from_nd2_path(nd2_path)
        metadata = Metadata(sample_metadata, image_metadata)
        return cls(intensities, metadata)

    @property
    def shape(self):
        """Get the shape of the intensity array."""
        return self.intensities.shape

    @property
    def channels(self):
        """Get the list of channels in this image."""
        return [channel_metadata.channel for channel_metadata in self.metadata.image.channels]

    @property
    def num_channels(self):
        """Get the number of channels in this image."""
        return len(self.channels)

    def get_intensities_from_channel(self, channel: Channel | str) -> UInt16Array:
        """Extract intensity data for a specific channel.

        Args:
            channel: The channel to extract, either as Channel enum or string name.

        Returns:
            UInt16Array: The intensity values for the specified channel.

        Raises:
            ValueError: If the specified channel is not in this image.
        """
        if isinstance(channel, str):
            channel = Channel[channel]

        if channel in self.channels:
            if self.intensities.ndim > 2:
                channel_index = self.channels.index(channel)
                return self.intensities[channel_index, :, :].copy()
            else:
                return self.intensities.copy()
        else:
            raise ValueError(f"No '{channel.name}' channel in image.")

    def get_brightfield_intensities(self) -> UInt16Array:
        """Extract intensity data for the brightfield channel."""
        return self.get_intensities_from_channel(Channel.BF)


def _extract_plane_from_text_info(
    text_info: str,
    plane_index: int = 1,
) -> str:
    """Extract a specific "Plane" from `nd2.ND2File.text_info` metadata.

    "Plane" is a section of metadata within `text_info["description"]` that includes relevant
    fields such as binning, exposure time, and laser power. The "Plane #" convention is only used
    for multi-channel ND2 files. For single-channel ND2 files we simply return all the text in
    `text_info["description"]`.
    """
    plane_regex = rf"Plane #{plane_index}:[\s\S]*?(?=Plane #\d|$)"
    plane_match = re.search(plane_regex, text_info["description"])
    if not plane_match:  # only one channel
        return text_info["description"]
    else:
        return plane_match.group(0)


def _extract_sample_from_text_info(
    text_info: str,
    sample_index: int = 1,
) -> str:
    """Extract a specific "Sample" from `nd2.ND2File.text_info` metadata.

    "Sample" is a section of metadata within `text_info["capturing"]` that includes relevant
    fields such as binning and exposure time (but not laser power). It is akin to "Plane" and
    easier to parse, but less comprehensive. The "Sample #" convention is only used
    for multi-channel ND2 files. For single-channel ND2 files we simply return all the text in
    `text_info["capturing"]`.
    """
    sample_regex = rf"Sample {sample_index}:[\s\S]*?(?=Sample \d|$)"
    sample_match = re.search(sample_regex, text_info["capturing"])
    if not sample_match:  # only one channel
        return text_info["capturing"]
    else:
        return sample_match.group(0)


def _parse_timestamp_from_text_info(text_info) -> datetime:
    """Parse timestamp from `nd2.ND2File.text_info`."""
    timestamp = text_info["date"]
    return datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")


def _parse_binning_from_sample(sample_text: str) -> str:
    """Parse binning from "Sample" section of `nd2.ND2File.text_info`."""
    binning = None
    for line in sample_text.splitlines():
        if "Binning" in line:
            binning = line.split(":")[1].strip()
    return binning


def _parse_exposure_time_from_sample(sample_text: str) -> float | None:
    """Parse exposure time from 'Sample' section of text_info and convert to milliseconds.

    Returns None if exposure time is not found in the text.
    """
    # Search for exposure time
    search_pattern = r"Exposure: (\d+(?:\.\d+)?) (\w+)"
    for line in sample_text.splitlines():
        if "Exposure" in line:
            exposure_time_match = re.search(search_pattern, line)

    # Convert to float if search was successful
    if exposure_time_match:
        exposure_time = float(exposure_time_match.group(1))
        unit = exposure_time_match.group(2)
    else:
        warnings.warn("Exposure time not found.", UserWarning, stacklevel=2)
        return None

    # Convert exposure time to milliseconds
    if unit == "s":
        return 1e3 * exposure_time
    elif unit == "ms":
        return exposure_time
    elif unit == "us":
        return 1e-3 * exposure_time
    # TODO: update with elif's if other units are ever found
    else:
        raise ValueError(f"Unknown unit for exposure time: {unit}.")


def _parse_laser_power_from_plane(plane_text: str) -> float | None:
    """Parse laser power from 'Plane' section of text_info.

    Returns None if laser power is not found in the text.
    """
    # Search for laser power
    search_pattern = r"Power:\s*(-?\d+(\.\d*)?|-?\.\d+)"
    for line in plane_text.splitlines():
        if "Power" in line:
            power_match = re.search(search_pattern, line)
    # Convert to float if search was successful
    if power_match:
        laser_power_pct = float(power_match.group(1))
    else:
        warnings.warn("Laser power not found.", UserWarning, stacklevel=2)
        return None
    # TODO: Convert laser power to mW
    return laser_power_pct
