from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import liffile
import nd2

from .channels import Channel
from .microscopy_utils import ChannelMetadata, DimensionFlags
from .typing import UInt16Array


@dataclass
class ImageMetadata:
    """Image metadata for a microscopy image.

    Contains metadata for all channels in the image.
    """

    sizes: dict[str, int]
    channel_metadata_list: list[ChannelMetadata]

    @property
    def channel_axis(self) -> int | None:
        """"""
        if "C" in self.sizes:
            return next(i for i, k in enumerate(self.sizes) if k == "C")

    @property
    def dimensions(self) -> DimensionFlags:
        """Derive dimension flags by combining from all channels."""
        if not self.channel_metadata_list:
            return DimensionFlags(0)

        _dimensions = DimensionFlags(0)
        for channel_metadata in self.channel_metadata_list:
            _dimensions |= channel_metadata.dimensions

        # Add MULTICHANNEL flag if there are multiple channels
        if len(self.channel_metadata_list) > 1:
            _dimensions |= DimensionFlags.MULTICHANNEL

        return _dimensions

    @classmethod
    def from_nd2_path(
        cls,
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
        from .nikon import create_image_metadata_from_nd2

        return create_image_metadata_from_nd2(nd2_path, channels)


@dataclass
class Metadata:
    """Combined metadata for a microscopy image of a sample.

    Contains both sample-specific metadata and image acquisition metadata.
    """

    image: ImageMetadata
    sample: dict[str, Any] | None = None


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
        nd2_path: Path,
        channels: list[Channel] | None = None,
        sample_metadata: dict[str, Any] | None = None,
    ) -> MicroscopyImage:
        """Create MicroscopyImage from a Nikon ND2 file.

        Args:
            nd2_path: Path to the Nikon ND2 file.
            channels: Optional list of Channel objects to override automatic channel detection.
                If not provided, channels are inferred from the ND2 file's optical configuration.
            sample_metadata: Optional dictionary containing sample-specific metadata.

        Returns:
            MicroscopyImage: A new microscopy image with intensity data and metadata.
        """
        intensities = nd2.imread(nd2_path)
        image_metadata = ImageMetadata.from_nd2_path(nd2_path, channels)
        metadata = Metadata(image_metadata, sample_metadata)
        return cls(intensities, metadata)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the intensity array."""
        return self.intensities.shape

    @property
    def channels(self) -> list[Channel]:
        """Get the list of channels in this image."""
        if self.metadata.image is None:
            raise ValueError("No image metadata available")
        return [
            channel_metadata.channel
            for channel_metadata in self.metadata.image.channel_metadata_list
        ]

    @property
    def num_channels(self) -> int:
        """Get the number of channels in this image."""
        return len(self.channels)

    # def get_intensities_from_channel(self, channel: Channel) -> UInt16Array:
    #     """Extract intensity data for a specific channel.

    #     Returns all data for the requested channel, preserving temporal and spatial
    #     dimensions (e.g., time-lapse or Z-stack).

    #     Args:
    #         channel: The channel to extract, either as Channel enum or string name.

    #     Returns:
    #         Intensity array for the specified channel. Shape depends on acquisition:
    #         - 2D single frame: (Y, X)
    #         - Time-lapse: (T, Y, X)
    #         - Z-stack: (Z, Y, X)
    #         - Multi-channel 2D: (Y, X)
    #         - Multi-channel time-lapse/Z-stack: (T, Y, X) or (Z, Y, X)

    #     Raises:
    #         ValueError: If the specified channel is not in this image.
    #     """
    #     if channel not in self.channels:
    #         raise ValueError(f"No '{channel.name}' channel in image.")

    #     # Single channel - return all data (may include T or Z dimensions)
    #     if self.num_channels == 1:
    #         return self.intensities.copy()

    #     # Multi-channel - extract the specific channel
    #     # Assumes first axis is channel dimension for multi-channel data
    #     # TODO: Parse metadata.image.dimensions to determine axis order
    #     channel_index = self.channels.index(channel)
    #     return self.intensities[channel_index, ...].copy()
