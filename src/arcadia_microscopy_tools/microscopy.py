from __future__ import annotations
import re
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
    "CY5": 640,
}
CHANNEL_EMISSION_PEAKS_NM = {
    "DAPI": 450,
    "FITC": 512,
    "TRITC": 595,
    "CY5": 665,
}
CHANNEL_COLORS_RGB = {
    "BF": Color(255, 255, 255),
    "DIC": Color(255, 255, 255),
    "DAPI": Color(0, 51, 255),
    "FITC": Color(7, 255, 0),
    "TRITC": Color(255, 191, 0),
    "CY5": Color(163, 0, 0),
}

NIKON_OPTICAL_CONFIGS_MAP = {
    "Mono": "BF",
    "DAPI": "DAPI",
    "FITC BP": "FITC",
    "TRITC BP": "TRITC",
    "GFP 488 nm": "FITC",
    "640 nm": "CY5",
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
    objective: str | None = None
    magnification: float | None = None
    numerical_aperture: float | None = None
    zoom: float | None = None
    binning: str | None = None
    exposure_time_ms: float | None = None
    period_ms: float | None = None
    duration_s: float | None = None
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
        period_ms = _parse_period_from_plane(plane_text)
        duration_s = _parse_duration_from_plane(plane_text)
        laser_power_pct = _parse_laser_power_from_plane(plane_text)

        return cls(
            channel=channel,
            timestamp=timestamp,
            height_px=nd2_channel.volume.voxelCount[0],
            width_px=nd2_channel.volume.voxelCount[1],
            thickness_px=nd2_channel.volume.voxelCount[2],
            pixel_size_um=nd2_channel.volume.axesCalibration[:2],
            z_step_size_um=nd2_channel.volume.axesCalibration[2],
            objective=nd2_channel.microscope.objectiveName,
            magnification=nd2_channel.microscope.objectiveMagnification,
            numerical_aperture=nd2_channel.microscope.objectiveNumericalAperture,
            zoom=nd2_channel.microscope.zoomMagnification,
            binning=binning,
            exposure_time_ms=exposure_time_ms,
            period_ms=period_ms,
            duration_s=duration_s,
            laser_power_pct=laser_power_pct,
        )


@dataclass
class ImageMetadata:
    """Metadata for a complete microscopy image.

    Contains metadata for all channels in the image.
    """

    dimensions: str
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
            text_info = nd2f.text_info

        # Collect channel metadata
        channel_metadatas = []
        for i in range(num_channels):
            channel_metadata = ChannelMetadata.from_nd2_path(
                nd2_path,
                channel_index=i,
            )
            channel_metadatas.append(channel_metadata)

        dimensions = _parse_dimensions_from_text_info(text_info)
        return cls(dimensions, channel_metadatas)


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

    Single-channel example:
    >>> with nd2.ND2File(nd2_path) as single_channel_nd2_file:
            text_info = single_channel_nd2_file.text_info
    >>> print(text_info)
        {'capturing': 'Fusion, SN:500651\r\n'.      <-- only channel
                      'Exposure: 2 ms\r\n'
                      'Binning: 4x4\r\n'
                      'Scan Mode: Standard\r\n'
                      'Temperature: -8.0°C\r\n'
                      'Denoise.ai OFF\r\n'
                      'Clarify.ai OFF',
        'date': '8/19/2025  10:06:25 AM',
        'description': 'Metadata:\r\n'
                       'Dimensions: T(316)\r\n'
                       ...
        }

    Multichannel example:
    >>> with nd2.ND2File(nd2_path) as multichannel_nd2_file:
            text_info = multichannel_nd2_file.text_info
    >>> print(text_info)
        {'capturing': 'Fusion, SN:500651\r\n'
                      'Sample 1:\r\n'                  <-- 1st channel
                      '  Exposure: 50 ms\r\n'
                      '  Binning: 1x1\r\n'
                      '  Scan Mode: Standard\r\n'
                      '  Temperature: -8.0°C\r\n'
                      '  Denoise.ai OFF\r\n'
                      '  Clarify.ai OFF\r\n'
                      'Sample 2:\r\n'                  <-- 2nd channel
                      '  Exposure: 300 ms\r\n'
                      '  Binning: 1x1\r\n'
                      '  Scan Mode: Ultra-quiet\r\n'
                      '  Temperature: -8.0°C\r\n'
                      '  Denoise.ai OFF\r\n'
                      '  Clarify.ai OFF',
        'date': '7/31/2025  3:30:59 PM',
        'description': 'Metadata:\r\n'
                       'Dimensions: λ(2)\r\n'
                       ...
        }
    """
    sample_regex = rf"Sample {sample_index}:[\s\S]*?(?=Sample \d|$)"
    sample_match = re.search(sample_regex, text_info["capturing"])
    if not sample_match:  # only one channel
        return text_info["capturing"]
    else:
        return sample_match.group(0)


def _extract_plane_from_text_info(
    text_info: str,
    plane_index: int = 1,
) -> str:
    """Extract a specific "Plane" from `nd2.ND2File.text_info` metadata.

    "Plane" is a section of metadata within `text_info["description"]` that includes relevant
    fields such as binning, exposure time, and laser power. The "Plane #" convention is only used
    for multi-channel ND2 files. For single-channel ND2 files we simply return all the text in
    `text_info["description"]`.

    Single-channel example:
    >>> with nd2.ND2File(nd2_path) as single_channel_nd2_file:
            text_info = single_channel_nd2_file.text_info
    >>> print(text_info)
        {...
        'date': '8/19/2025  10:06:25 AM',
        'description': 'Metadata:\r\n'                 <-- only channel
                        'Dimensions: T(316)\r\n'
                        'Camera Name: Fusion, SN:500651\r\n'
                        'Numerical Aperture: 0.2\r\n'
                        'Refractive Index: 1\r\n'
                        ' Name: Mono\r\n'
                        ' Component Count: 1\r\n'
                        ' Modality: Widefield Fluorescence\r\n'
                        ' Camera Settings:   Exposure: 2 ms\r\n'
                        '  Binning: 4x4\r\n'
                        ...
                        'Time Loop: 316\r\n'
                        '- Equidistant (Period 50 ms)',
        'optics': 'PLAN APO λD 4x OFN25'}

    Multichannel example:
    >>> with nd2.ND2File(nd2_path) as multichannel_nd2_file:
            text_info = multichannel_nd2_file.text_info
    >>> print(text_info)
        {...
        'date': '7/31/2025  3:30:59 PM',
        'description': 'Metadata:\r\n'
                        'Dimensions: λ(2)\r\n'
                        'Camera Name: Fusion, SN:500651\r\n'
                        'Numerical Aperture: 0.75\r\n'
                        'Refractive Index: 1\r\n'
                        'Number of Picture Planes: 2\r\n'
                        'Plane #1:\r\n'                <-- 1st channel
                        ' Name: Mono\r\n'
                        ' Component Count: 1\r\n'
                        ' Modality: Widefield Fluorescence, Spinning Disk Confocal\r\n'
                        ' Camera Settings:   Exposure: 50 ms\r\n'
                        '  Binning: 1x1\r\n'
                        ...
                        'Plane #2:\r\n'                <-- 2nd channel
                        ' Name: DAPI\r\n'
                        ' Component Count: 1\r\n'
                        ' Modality: Widefield Fluorescence, Spinning Disk Confocal\r\n'
                        ' Camera Settings:   Exposure: 300 ms\r\n'
                        '  Binning: 1x1\r\n'
                        ...
        'optics': 'Plan Apo λ 20x'}
    """
    plane_regex = rf"Plane #{plane_index}:[\s\S]*?(?=Plane #\d|$)"
    plane_match = re.search(plane_regex, text_info["description"])
    if not plane_match:  # only one channel
        return text_info["description"]
    else:
        return plane_match.group(0)


def _parse_timestamp_from_text_info(text_info: str) -> datetime:
    """Parse timestamp from `nd2.ND2File.text_info`."""
    timestamp = text_info["date"]
    return datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")


def _parse_dimensions_from_text_info(text_info: str) -> str:
    """"""
    dimensions = ""
    description = text_info["description"]
    for line in description.splitlines():
        if "Dimensions" in line:
            dimensions = line.split(":")[1].strip()
    return dimensions


def _parse_binning_from_sample(sample_text: str) -> str:
    """Parse binning from "Sample" section of `nd2.ND2File.text_info`."""
    binning = ""
    for line in sample_text.splitlines():
        if "Binning" in line:
            binning = line.split(":")[1].strip()
    return binning


def _parse_exposure_time_from_sample(sample_text: str) -> float | None:
    """Parse exposure time from 'Plane' section of text_info.

    Returns None if the exposure time is not found in the text."""
    exposure_time_ms = None
    pattern = r"Exposure: (\d+(?:\.\d+)?) (\w+)"
    for line in sample_text.splitlines():
        if "Exposure" in line:
            match = re.search(pattern, line)
            if match:
                time, unit = match.groups()
                exposure_time_ms = _convert_time_to_ms(time, unit)
                break
    return exposure_time_ms


def _parse_period_from_plane(plane_text: str) -> float | None:
    """Parse period from 'Plane' section of text_info.

    The period (or frame interval) is the time between the start of one frame and the start of the
    next in a sequence. It's distinct from the exposure time, which is the portion of that period
    during which the sample is illuminated and the camera is actively collecting light. Depending
    on the acquisition settings of the timelapse set in the software, the ND2 file metadata will
    either output period or duration but not both.

    Returns None if the period is not found in the text.
    """
    period_ms = None
    pattern = r"Period\s+(\d+(?:\.\d+)?)\s*(\w+)"
    for line in plane_text.splitlines():
        if "Period" in line:
            match = re.search(pattern, line)
            if match:
                time, unit = match.groups()
                period_ms = _convert_time_to_ms(time, unit)
                break
    return period_ms


def _parse_duration_from_plane(plane_text: str) -> float | None:
    """Parse duration from 'Plane' section of text_info.

    The duration is the total time of a timelapse. Depending on the acquisition settings of the
    timelapse set in the software, the ND2 file metadata will either output period or duration
    but not both.

    Returns None if the duration is not found in the text.
    """
    duration_s = None
    pattern = r"Duration\s+(\d+(?:\.\d+)?)\s*(\w+)"
    for line in plane_text.splitlines():
        if "Duration" in line:
            match = re.search(pattern, line)
            if match:
                time, unit = match.groups()
                duration_s = _convert_time_to_ms(time, unit) / 1000
                break
    return duration_s


def _parse_laser_power_from_plane(plane_text: str) -> float | None:
    """Parse laser power from 'Plane' section of text_info.

    Returns None if laser power is not found in the text.
    """
    laser_power_pct = None
    pattern = r"Power:\s*(-?\d+(\.\d*)?|-?\.\d+)"
    for line in plane_text.splitlines():
        if "Power" in line:
            match = re.search(pattern, line)
            if match:
                laser_power_pct = float(match.group(1))
    # TODO: convert % to mW
    return laser_power_pct


def _convert_time_to_ms(time: str, unit: str) -> float:
    """Converts time to milliseconds."""
    time = float(time)
    if "h" in unit:
        return 3600 * 1000 * time
    elif unit == "min":
        return 60 * 1e3 * time
    elif unit == "s":
        return 1e3 * time
    elif unit == "ms":
        return time
    elif unit == "us" or unit == "µs":
        return 1e-3 * time
    else:
        raise ValueError(f"Unknown unit of time: {unit}.")
