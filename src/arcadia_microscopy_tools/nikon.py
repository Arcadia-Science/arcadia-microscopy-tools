from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

import nd2
from nd2.structures import TextInfo

from .channels import Channel
from .microscopy import ImageMetadata
from .microscopy_utils import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MicroscopeSettings,
    PhysicalDimensions,
)


def create_image_metadata_from_nd2(
    nd2_path: Path,
    channels: list[Channel] | None = None,
) -> ImageMetadata:
    """Create ImageMetadata from a Nikon ND2 file.

    Args:
        nd2_path: Path to the Nikon ND2 file.
        channels: Optional list of Channel objects to override automatic channel detection.
            If not provided, channels are inferred from the ND2 file's optical configuration.

    Returns:
        ImageMetadata with sizes and channel metadata for all channels.
    """
    parser = _NikonMetadataParser(nd2_path, channels)
    return parser.parse()


class _NikonMetadataParser:
    """Parser for extracting metadata from Nikon ND2 files."""

    def __init__(self, nd2_path: Path, channels: list[Channel] | None = None):
        self.nd2_path = nd2_path
        self.channels = channels
        self._nd2f: nd2.ND2File
        self._text_info: TextInfo

    def parse(self) -> ImageMetadata:
        """Parse the ND2 file and extract all metadata."""
        with nd2.ND2File(self.nd2_path) as self._nd2f:
            self._text_info = TextInfo(self._nd2f.text_info)

            sizes = dict(self._nd2f.sizes)
            channel_metadata_list = self._parse_all_channels()

            return ImageMetadata(sizes=sizes, channel_metadata_list=channel_metadata_list)

    def _parse_all_channels(self) -> list[ChannelMetadata]:
        """Parse metadata for all channels in the ND2 file."""
        if self._nd2f.metadata.contents is None:
            raise ValueError(f"No metadata contents available in {self.nd2_path}")

        num_channels = self._nd2f.metadata.contents.channelCount

        # Validate channels list length if provided
        if self.channels is not None and len(self.channels) != num_channels:
            raise ValueError(
                f"Expected {num_channels} channels but got {len(self.channels)} in channels list"
            )

        channel_metadata_list = []

        for i in range(num_channels):
            channel = self.channels[i] if self.channels else None
            channel_metadata = self._parse_channel_metadata(i, channel)
            channel_metadata_list.append(channel_metadata)

        return channel_metadata_list

    def _parse_channel_metadata(
        self,
        channel_index: int,
        channel: Channel | None = None,
    ) -> ChannelMetadata:
        """Parse metadata for a specific channel."""
        nd2_channel = self._get_nd2_channel_metadata(channel_index)

        if channel is None:
            channel = Channel.from_optical_config_name(nd2_channel.channel.name)

        dimensions = self._get_dimension_flags()
        timestamp = self._parse_timestamp()
        resolution = self._parse_physical_dimensions(nd2_channel, dimensions)
        acquisition = self._parse_acquisition_settings(nd2_channel, channel_index, dimensions)
        optics = self._parse_microscope_settings(nd2_channel)

        return ChannelMetadata(
            channel=channel,
            timestamp=timestamp,
            dimensions=dimensions,
            resolution=resolution,
            acquisition=acquisition,
            optics=optics,
        )

    def _get_nd2_channel_metadata(self, channel_index: int):
        """Get the nd2 channel metadata object for a specific channel."""
        channels = self._nd2f.metadata.channels
        if channels is None:
            raise ValueError("No channel metadata available")
        return channels[channel_index]

    def _get_dimension_flags(self) -> DimensionFlags:
        """Determine dimension flags from ND2 file sizes for a single channel."""
        dimensions = DimensionFlags(0)
        sizes = self._nd2f.sizes

        if "T" in sizes and sizes["T"] > 1:
            dimensions |= DimensionFlags.TIMELAPSE

        if "Z" in sizes and sizes["Z"] > 1:
            dimensions |= DimensionFlags.Z_STACK

        if "S" in sizes and sizes["S"] > 1:
            dimensions |= DimensionFlags.RGB

        # TODO: Add checks for SPECTRAL, MONTAGE
        # MONTAGE might be "XY"

        return dimensions

    def _parse_timestamp(self) -> datetime:
        """Parse timestamp from text_info."""
        if "date" not in self._text_info:
            raise ValueError("Missing 'date' field in text_info")

        timestamp = self._text_info["date"]
        return datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")

    def _parse_physical_dimensions(
        self,
        nd2_channel: nd2.structures.Channel,
        dimensions: DimensionFlags,
    ) -> PhysicalDimensions:
        """Parse physical dimensions from nd2 channel metadata."""
        pixel_size_um_x, pixel_size_um_y, _ = nd2_channel.volume.axesCalibration
        return PhysicalDimensions(
            height_px=nd2_channel.volume.voxelCount[0],
            width_px=nd2_channel.volume.voxelCount[1],
            pixel_size_um=(pixel_size_um_x + pixel_size_um_y) / 2,
            thickness_px=nd2_channel.volume.voxelCount[2] if dimensions.is_zstack else None,
            z_step_size_um=nd2_channel.volume.axesCalibration[2] if dimensions.is_zstack else None,
        )

    def _parse_acquisition_settings(
        self,
        nd2_channel: nd2.structures.Channel,
        channel_index: int,
        dimensions: DimensionFlags,
    ) -> AcquisitionSettings:
        """Parse acquisition settings from nd2 channel metadata and text_info."""
        sample_text = self._extract_sample_text(channel_index)
        binning = self._parse_binning(sample_text)
        exposure_time_ms = self._parse_exposure_time(sample_text)

        frame_interval_ms = None
        if dimensions.is_timelapse:
            frame_interval_ms = self._parse_frame_interval()
            # If periodMs is 0 or missing, try to calculate from duration and frame count
            if frame_interval_ms is None or frame_interval_ms == 0:
                duration_ms = self._parse_duration()
                if duration_ms and "T" in self._nd2f.sizes:
                    num_frames = self._nd2f.sizes["T"]
                    if num_frames > 1:
                        frame_interval_ms = duration_ms / num_frames

        return AcquisitionSettings(
            exposure_time_ms=exposure_time_ms or 0.0,
            zoom=nd2_channel.microscope.zoomMagnification,
            binning=binning,
            frame_interval_ms=frame_interval_ms,
            wavelengths_nm=None,  # TODO: extract wavelengths for spectral data
        )

    def _parse_microscope_settings(self, nd2_channel: nd2.structures.Channel) -> MicroscopeSettings:
        """Parse microscope settings from nd2 channel metadata."""
        magnification = nd2_channel.microscope.objectiveMagnification
        numerical_aperture = nd2_channel.microscope.objectiveNumericalAperture

        return MicroscopeSettings(
            magnification=int(magnification) if magnification is not None else 0,
            numerical_aperture=numerical_aperture or 0.0,
            objective=nd2_channel.microscope.objectiveName,
            light_source=None,  # TODO: extract light source info
            laser_power_mw=None,  # TODO: convert laser_power_pct to mW
        )

    def _extract_sample_text(self, channel_index: int) -> str:
        """Extract 'Sample' section from text_info for a specific channel.

        For single-channel files, returns entire 'capturing' field.
        For multi-channel files, extracts specific 'Sample N:' section.
        """
        if "capturing" not in self._text_info:
            raise ValueError("Missing 'capturing' field in text_info")

        sample_index = channel_index + 1
        sample_regex = rf"Sample {sample_index}:[\s\S]*?(?=Sample \d|$)"
        sample_match = re.search(sample_regex, self._text_info["capturing"])

        if not sample_match:
            return self._text_info["capturing"]
        else:
            return sample_match.group(0)

    def _extract_plane_text(self, channel_index: int) -> str:
        """Extract 'Plane' section from text_info for a specific channel.

        For single-channel files, returns entire 'description' field.
        For multi-channel files, extracts specific 'Plane #N:' section.
        """
        if "description" not in self._text_info:
            raise ValueError("Missing 'description' field in text_info")

        plane_index = channel_index + 1
        plane_regex = rf"Plane #{plane_index}:[\s\S]*?(?=Plane #\d|$)"
        plane_match = re.search(plane_regex, self._text_info["description"])

        if not plane_match:
            return self._text_info["description"]
        else:
            return plane_match.group(0)

    def _parse_binning(self, sample_text: str) -> str | None:
        """Parse binning from sample text."""
        for line in sample_text.splitlines():
            if "Binning" in line:
                return line.split(":")[1].strip()
        return None

    def _parse_exposure_time(self, sample_text: str) -> float | None:
        """Parse exposure time from sample text, converting to milliseconds."""
        pattern = r"Exposure: (\d+(?:\.\d+)?) (\w+)"
        for line in sample_text.splitlines():
            if "Exposure" in line:
                match = re.search(pattern, line)
                if match:
                    time, unit = match.groups()
                    return self._convert_time_to_ms(time, unit)
        return None

    def _parse_frame_interval(self) -> float | None:
        """Parse frame interval (period) from experiment metadata."""
        if self._nd2f.experiment:
            for loop in self._nd2f.experiment:
                if loop.type == "TimeLoop":
                    return loop.parameters.periodMs
        return None

    def _parse_duration(self) -> float | None:
        """Parse total duration from experiment metadata."""
        if self._nd2f.experiment:
            for loop in self._nd2f.experiment:
                if loop.type == "TimeLoop":
                    return loop.parameters.durationMs
        return None

    def _parse_laser_power(self, plane_text: str) -> float | None:
        """Parse laser power percentage from plane text."""
        pattern = r"Power:\s*(-?\d+(\.\d*)?|-?\.\d+)"
        for line in plane_text.splitlines():
            if "Power" in line:
                match = re.search(pattern, line)
                if match:
                    return float(match.group(1))
        return None

    @staticmethod
    def _convert_time_to_ms(time: str | float, unit: str) -> float:
        """Convert time to milliseconds from various units."""
        time_value = float(time)
        if "h" in unit:
            return 3600 * 1000 * time_value
        elif unit == "min":
            return 60 * 1000 * time_value
        elif unit == "s":
            return 1000 * time_value
        elif unit == "ms":
            return time_value
        elif unit in ("us", "µs"):
            return 0.001 * time_value
        else:
            raise ValueError(f"Unknown unit of time: {unit}")
