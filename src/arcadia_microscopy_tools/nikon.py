from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

import nd2
import pandas as pd
from nd2.structures import TextInfo

from .channels import Channel
from .metadata_structures import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MeasuredDimensions,
    MicroscopeConfig,
    NominalDimensions,
)
from .microscopy import InstrumentMetadata
from .typing import Float64Array


def create_instrument_metadata_from_nd2(
    nd2_path: Path,
    channels: list[Channel] | None = None,
) -> InstrumentMetadata:
    """Create InstrumentMetadata from a Nikon ND2 file.

    Args:
        nd2_path: Path to the Nikon ND2 file.
        channels: Optional list of Channel objects to override automatic channel detection.
            If not provided, channels are inferred from the ND2 file's optical configuration.

    Returns:
        InstrumentMetadata with sizes and channel metadata for all channels.
    """
    parser = _NikonMetadataParser(nd2_path, channels)
    return parser.parse()


class _NikonMetadataParser:
    """Parser for extracting metadata from Nikon ND2 files."""

    def __init__(self, nd2_path: Path, channels: list[Channel] | None = None):
        self.nd2_path = nd2_path
        self.channels = channels
        self._nd2f: nd2.ND2File

    def parse(self) -> InstrumentMetadata:
        """Parse the ND2 file and extract all metadata."""
        with nd2.ND2File(self.nd2_path) as self._nd2f:
            self.sizes = dict(self._nd2f.sizes)
            self.text_info = TextInfo(self._nd2f.text_info)
            self.events = self._nd2f.events()
            self.dimensions = self._get_dimension_flags()
            self.timestamp = self._parse_timestamp()

            channel_metadata_list = self._parse_all_channels()

        return InstrumentMetadata(self.sizes, channel_metadata_list)

    def _parse_all_channels(self) -> list[ChannelMetadata]:
        """Parse metadata for all channels in the ND2 file."""
        if self._nd2f.metadata.contents is None:
            raise ValueError(f"No metadata contents available in {self.nd2_path}")

        # Validate channels list length if provided
        num_channels = self._nd2f.metadata.contents.channelCount
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

        resolution = self._parse_nominal_dimensions(nd2_channel)
        measured = self._parse_measured_dimensions()
        acquisition = self._parse_acquisition_settings(nd2_channel, channel_index)
        optics = self._parse_microscope_settings(nd2_channel)

        return ChannelMetadata(
            channel=channel,
            timestamp=self.timestamp,
            dimensions=self.dimensions,
            resolution=resolution,
            measured=measured,
            acquisition=acquisition,
            optics=optics,
        )

    def _get_nd2_channel_metadata(self, channel_index: int) -> nd2.structures.Channel:
        """Get the nd2 channel metadata object for a specific channel."""
        channels = self._nd2f.metadata.channels
        if channels is None:
            raise ValueError("No channel metadata available")
        return channels[channel_index]

    def _get_dimension_flags(self) -> DimensionFlags:
        """Determine dimension flags from ND2 file sizes for a single channel."""
        dimensions = DimensionFlags(0)

        if "T" in self.sizes and self.sizes["T"] > 1:
            dimensions |= DimensionFlags.TIMELAPSE
        if "Z" in self.sizes and self.sizes["Z"] > 1:
            dimensions |= DimensionFlags.Z_STACK
        if "S" in self.sizes and self.sizes["S"] > 1:
            dimensions |= DimensionFlags.RGB
        if "P" in self.sizes and self.sizes["P"] > 1:
            dimensions |= DimensionFlags.MONTAGE

        return dimensions

    def _parse_timestamp(self) -> datetime:
        """Parse timestamp from text_info."""
        if "date" not in self.text_info:
            raise ValueError("Missing 'date' field in text_info")

        timestamp = self.text_info["date"]
        return datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")

    def _parse_nominal_dimensions(self, nd2_channel: nd2.structures.Channel) -> NominalDimensions:
        """Parse nominal dimensions from nd2 channel metadata."""
        # Spatial dimensions
        x_size_px, y_size_px, z_size_px = nd2_channel.volume.voxelCount
        x_step_um, y_step_um, z_step_um = nd2_channel.volume.axesCalibration
        xy_step_um = (x_step_um + y_step_um) / 2

        # Time dimension - extract from events if available
        t_size_px = None
        t_step_ms = None
        if self.events:
            t_size_px = self.sizes.get("T")
            t_step_ms = self.events[0].get("Exposure Time [ms]")

        return NominalDimensions(
            x_size_px=x_size_px,
            y_size_px=y_size_px,
            xy_step_um=xy_step_um,
            z_size_px=z_size_px if self.dimensions.is_zstack else None,
            z_step_um=z_step_um if self.dimensions.is_zstack else None,
            t_size_px=t_size_px if self.dimensions.is_timelapse else None,
            t_step_ms=t_step_ms if self.dimensions.is_timelapse else None,
            w_size_px=None,
            w_step_nm=None,
        )

    def _parse_measured_dimensions(self) -> MeasuredDimensions:
        """Parse measured dimension values from event metadata.

        Extracts actual z-positions, frame intervals, and wavelengths from the
        events metadata rather than nominal values.
        """
        x_values_um = None
        y_values_um = None
        z_values_um = None
        t_values_ms = None
        w_values_nm = None

        events_dataframe = pd.DataFrame(self.events)

        if len(events_dataframe) < 2:
            return MeasuredDimensions(
                z_values_um=z_values_um,
                t_values_ms=t_values_ms,
                w_values_nm=w_values_nm,
            )

        if self.dimensions.is_montage:
            x_values_um, y_values_um = self._extract_xy_coordinates(events_dataframe)

        if self.dimensions.is_zstack:
            z_values_um = self._extract_z_coordinates(events_dataframe)

        if self.dimensions.is_timelapse:
            t_values_ms = self._extract_time_coordinates(events_dataframe)

        if self.dimensions.is_spectral:
            w_values_nm = self._extract_wavelength_coordinates(events_dataframe)

        return MeasuredDimensions(
            x_values_um=x_values_um,
            y_values_um=y_values_um,
            z_values_um=z_values_um,
            t_values_ms=t_values_ms,
            w_values_nm=w_values_nm,
        )

    def _extract_xy_coordinates(
        self, events_dataframe: pd.DataFrame
    ) -> tuple[Float64Array, Float64Array]:
        """Extract stage coordinates from events for tiled imaging.

        Not yet implemented - stage position data extraction needs additional testing.
        """
        raise NotImplementedError(
            "(X, Y) position extraction for tiled imaging is not yet implemented"
        )

    def _extract_z_coordinates(self, events_dataframe: pd.DataFrame) -> Float64Array:
        """Extract z-coordinates from events, centered around z=0.

        Dynamically selects the appropriate z-column based on which has variation
        (different hardware configurations use different column names).
        """
        z_columns = ["Z Coord [µm]", "Ti2 ZDrive [µm]", "NIDAQ Piezo Z (name: Piezo Z) [µm]"]

        # Find which z column exists and has variation
        dynamic_z_column = None
        for z_col in z_columns:
            if z_col in events_dataframe.columns and events_dataframe[z_col].nunique() > 1:
                dynamic_z_column = z_col
                break

        if dynamic_z_column is None:
            raise ValueError("No varying Z coordinate column found in events")

        if "Z-Series" not in events_dataframe.columns:
            raise ValueError("Missing 'Z-Series' column in events metadata")

        z_values_um = events_dataframe[dynamic_z_column].to_numpy(dtype=float)
        z_center_index = events_dataframe["Z-Series"].abs().idxmin()
        z_center = events_dataframe.loc[z_center_index, dynamic_z_column]
        z_values_um -= z_center

        return z_values_um

    def _extract_time_coordinates(self, events_dataframe: pd.DataFrame) -> Float64Array:
        """Extract time coordinates from events, relative to the first frame.

        Returns time values in milliseconds, zeroed to the start of acquisition.
        """
        if "Time [s]" not in events_dataframe.columns:
            raise ValueError("Missing 'Time [s]' column in events metadata")

        t_values_s = events_dataframe["Time [s]"].to_numpy(dtype=float)
        t_values_ms = 1e3 * (t_values_s - t_values_s.min())
        return t_values_ms

    def _extract_wavelength_coordinates(self, events_dataframe: pd.DataFrame) -> Float64Array:
        """Extract wavelength coordinates from events for spectral imaging.

        Not yet implemented - spectral data extraction needs additional testing.
        """
        raise NotImplementedError(
            "Wavelength extraction for spectral imaging is not yet implemented"
        )

    def _parse_acquisition_settings(
        self,
        nd2_channel: nd2.structures.Channel,
        channel_index: int,
    ) -> AcquisitionSettings:
        """Parse acquisition settings from nd2 channel metadata and text_info."""
        sample_text = self._extract_sample_text(channel_index)
        exposure_time_s = self._parse_exposure_time(sample_text)
        zoom = nd2_channel.microscope.zoomMagnification
        binning = self._parse_binning(sample_text)

        return AcquisitionSettings(
            exposure_time_s=exposure_time_s,
            zoom=zoom,
            binning=binning,
            pixel_dwell_time_us=None,
            line_scan_speed_hz=None,
            line_averaging=None,
            line_accumulation=None,
            frame_averaging=None,
            frame_accumulation=None,
        )

    def _parse_microscope_settings(self, nd2_channel: nd2.structures.Channel) -> MicroscopeConfig:
        """Parse microscope settings from nd2 channel metadata."""
        magnification = nd2_channel.microscope.objectiveMagnification
        numerical_aperture = nd2_channel.microscope.objectiveNumericalAperture

        return MicroscopeConfig(
            magnification=int(magnification) if magnification is not None else 0,
            numerical_aperture=numerical_aperture or 0.0,
            objective=nd2_channel.microscope.objectiveName,
            light_source=None,  # TODO: extract light source info
            power_mw=None,  # TODO: convert power_pct to mW
        )

    def _extract_sample_text(self, channel_index: int) -> str:
        """Extract 'Sample' section from text_info for a specific channel.

        For single-channel files, returns entire 'capturing' field.
        For multi-channel files, extracts specific 'Sample N:' section.
        """
        if "capturing" not in self.text_info:
            raise ValueError("Missing 'capturing' field in text_info")

        sample_index = channel_index + 1
        sample_regex = rf"Sample {sample_index}:[\s\S]*?(?=Sample \d|$)"
        sample_match = re.search(sample_regex, self.text_info["capturing"])

        return sample_match.group(0) if sample_match else self.text_info["capturing"]

    def _extract_plane_text(self, channel_index: int) -> str:
        """Extract 'Plane' section from text_info for a specific channel.

        For single-channel files, returns entire 'description' field.
        For multi-channel files, extracts specific 'Plane #N:' section.
        """
        if "description" not in self.text_info:
            raise ValueError("Missing 'description' field in text_info")

        plane_index = channel_index + 1
        plane_regex = rf"Plane #{plane_index}:[\s\S]*?(?=Plane #\d|$)"
        plane_match = re.search(plane_regex, self.text_info["description"])

        return plane_match.group(0) if plane_match else self.text_info["description"]

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
                    time_s = self._convert_time_to_s(time, unit)
                    return time_s
        return None

    def _parse_power(self, plane_text: str) -> float | None:
        """Parse laser power percentage from plane text."""

        # Parsing the power is tricky:
        #     1. Units are percentages and total power is unknown
        #     2. Not trivial to determine the light source for non-fluorescence
        #        channels (e.g. BRIGHTFIELD, DIC)

        #     Example snippet of plane_text from tests/data/example-multichannel.nd2:
        #         Plane #1:
        #             Name: Mono
        #             Component Count: 1
        #             Modality: Widefield Fluorescence
        #             Camera Settings:   Exposure: 20 ms
        #             ...
        #             LUN-F, MultiLaser(LUN-F):
        #                 Line:3; ExW:561; Power: 82.5; On

        #             Lida, Shutter(Lida): Active
        #             Lida, MultiLaser(Lida):
        #                 Line:1; ExW:450; Power:  5.0; On
        #                 Line:2; ExW:550; Power:  5.0; On
        #                 Line:3; ExW:640; Power:  5.0; On

        pattern = r"Power:\s*(-?\d+(\.\d*)?|-?\.\d+)"
        for line in plane_text.splitlines():
            if "Power" in line:
                match = re.search(pattern, line)
                if match:
                    return float(match.group(1))
        return None

    @staticmethod
    def _convert_time_to_s(time: str | float, unit: str) -> float:
        """Convert time to seconds from various units."""
        time_value = float(time)
        if "h" in unit:
            return 3600 * time_value
        elif unit == "min":
            return 60 * time_value
        elif unit == "s":
            return time_value
        elif unit == "ms":
            return time_value / 1000
        elif unit in ("us", "µs"):
            return time_value / 1_000_000
        else:
            raise ValueError(f"Unknown unit of time: {unit}")
