from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import liffile
import nd2

from .channels import Channel
from .metadata_structures import ChannelMetadata, DimensionFlags
from .pipeline import Pipeline, PipelineParallelized
from .typing import ScalarArray, UInt16Array


@dataclass
class ImageMetadata:
    """Image metadata for a microscopy image.

    Contains metadata for all channels in the image.

    Attributes:
        sizes: Mapping of dimensions to sizes (e.g., {'T': 100, 'C': 2, 'Y': 512, 'X': 512}).
        channel_metadata_list: List of ChannelMetadata objects for each channel in the image.
        channel_axis: Axis index for the channel dimension, or None if single channel.
        dimensions: Dimension flags indicating which dimensions are present in the image.
    """

    sizes: dict[str, int]
    channel_metadata_list: list[ChannelMetadata]

    @property
    def channel_axis(self) -> int | None:
        """Get the axis index for the channel dimension, or None if single channel."""
        if "C" in self.sizes:
            return next((i for i, k in enumerate(self.sizes.keys()) if k == "C"), None)

    @cached_property
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

    Attributes:
        image: Image acquisition metadata including dimensions and channel information.
        sample: Optional dictionary containing sample-specific metadata.
    """

    image: ImageMetadata
    sample: dict[str, Any] | None = None


@dataclass
class MicroscopyImage:
    """Dataclass for microscopy image data.

    Contains both the image intensity data and associated metadata for all channels.
    Provides methods to access specific channel data.

    Attributes:
        intensities: NumPy array containing the image intensity data. Shape depends on
            acquisition type (e.g., (Y, X) for 2D, (T, Y, X) for time-lapse,
            (T, C, Y, X) for multi-channel time-lapse).
        metadata: Combined metadata containing image acquisition metadata and optional
            sample-specific metadata.
        shape: The shape of the intensity array.
        sizes: Dimension sizes dictionary (e.g., {'T': 100, 'C': 2, 'Y': 512, 'X': 512}).
        dimensions: Dimension flags indicating which dimensions are present in the image.
        channels: List of Channel objects present in this image.
        channel_axis: Axis index for the channel dimension, or None if single channel.
        num_channels: Number of channels in this image.
    """

    intensities: UInt16Array
    metadata: Metadata

    def __repr__(self) -> str:
        """Return a concise string representation of the microscopy image."""
        dtype_str = f"dtype={self.intensities.dtype}"

        # Show first few and last few intensity values
        flat = self.intensities.flatten()
        if len(flat) <= 10:
            intensity_str = f"intensities={list(flat)}"
        else:
            first_vals = flat[:3].tolist()
            last_vals = flat[-3:].tolist()
            intensity_str = (
                f"intensities=[{', '.join(map(str, first_vals))}, ..., "
                f"{', '.join(map(str, last_vals))}]"
            )

        # Add dimension/channel info if available
        try:
            sizes_str = f"sizes={self.sizes}"
            channels_str = f"channels={[channel.name for channel in self.channels]}"
            info = f"{sizes_str}, {channels_str}, {intensity_str}, {dtype_str}"
        except ValueError:
            info = f"{intensity_str}, {dtype_str}"

        return f"MicroscopyImage({info})"

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

    @classmethod
    def from_lif_path(
        cls,
        lif_path: Path,
        image_name: str,
        sample_metadata: dict[str, Any] | None = None,
    ) -> MicroscopyImage:
        """Create MicroscopyImage from a Leica LIF file.

        Args:
            lif_path: Path to the Leica LIF file.
            image_name: Name of the image within the LIF file to load.
            sample_metadata: Optional dictionary containing sample-specific metadata.

        Returns:
            MicroscopyImage: A new microscopy image with intensity data and metadata.

        Note:
            LIF files currently have minimal metadata support. Channel metadata is not
            parsed, so operations requiring channel information may not work as expected.
        """
        with liffile.LifFile(lif_path) as lif:
            for image in lif.images:
                if image.name == image_name:
                    intensities = image.asarray()
                    sizes = image.sizes
                    break
            else:
                raise ValueError(
                    f"Image {image_name} not found in {lif_path}. Available images: "
                    f"{[image.name for image in lif.images]}"
                )

        # TODO: create parser for LIF metadata - for now create minimal ImageMetadata
        image_metadata = ImageMetadata(sizes=sizes, channel_metadata_list=[])
        metadata = Metadata(image_metadata, sample_metadata)
        return cls(intensities, metadata)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the intensity array."""
        return self.intensities.shape

    @property
    def sizes(self) -> dict[str, int]:
        """Get the dimension sizes dictionary (e.g., {'T': 100, 'C': 2, 'Y': 512, 'X': 512})."""
        return self.metadata.image.sizes

    @property
    def dimensions(self) -> DimensionFlags:
        """Get the dimension flags indicating which dimensions are present."""
        return self.metadata.image.dimensions

    @property
    def channels(self) -> list[Channel]:
        """Get the list of channels in this image."""
        return [
            channel_metadata.channel
            for channel_metadata in self.metadata.image.channel_metadata_list
        ]

    @property
    def channel_axis(self) -> int | None:
        """Get the axis index for the channel dimension, or None if single channel."""
        return self.metadata.image.channel_axis

    @property
    def num_channels(self) -> int:
        """Get the number of channels in this image."""
        return len(self.metadata.image.channel_metadata_list)

    def get_intensities_from_channel(self, channel: Channel) -> UInt16Array:
        """Extract intensity data for a specific channel.

        Returns all data for the requested channel, preserving temporal and spatial
        dimensions (e.g., time-lapse or Z-stack).

        Args:
            channel: The Channel object to extract.

        Returns:
            Intensity array for the specified channel. Shape depends on acquisition:
            - 2D single frame: (Y, X)
            - Time-lapse: (T, Y, X)
            - Z-stack: (Z, Y, X)
            - Multi-channel 2D: (Y, X)
            - Multi-channel time-lapse/Z-stack: (T, Y, X) or (Z, Y, X)

        Raises:
            ValueError: If the specified channel is not in this image or no metadata available.
        """
        if channel not in self.channels:
            raise ValueError(
                f"Channel '{channel.name}' not found in image. Available channels: "
                f"{[channel.name for channel in self.channels]}"
            )

        # Single channel - return all data (may include T or Z dimensions)
        if self.num_channels == 1:
            return self.intensities.copy()

        # Multi-channel - extract the specific channel using channel_axis
        channel_index = self.channels.index(channel)
        if self.channel_axis is None:
            raise ValueError("Channel axis not found in metadata")

        # Build slice tuple to extract the channel
        slices: list[slice | int] = [slice(None)] * len(self.intensities.shape)
        slices[self.channel_axis] = channel_index

        return self.intensities[tuple(slices)].copy()

    def apply_pipeline(
        self,
        pipeline: Pipeline | PipelineParallelized,
        channel: Channel,
    ) -> ScalarArray:
        """Apply a processing pipeline to intensity data from a specific channel.

        Extracts the intensity data for the specified channel and processes it
        through the provided pipeline. Supports both standard and parallelized pipelines.

        Args:
            pipeline: The processing pipeline to apply. Can be either a Pipeline or
                PipelineParallelized instance containing the sequence of transformations.
            channel: The Channel object whose intensity data should be processed.

        Returns:
            Processed intensity data as a scalar array. The shape and dtype depend on
            the specific transformations in the pipeline.

        Raises:
            ValueError: If the specified channel is not found in this image or if no
                image metadata is available.
        """
        intensities = self.get_intensities_from_channel(channel)
        return pipeline(intensities)
