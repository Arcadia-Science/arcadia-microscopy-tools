from __future__ import annotations
import os
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from .mask_processing import SegmentationMask
from .microscopy import Channel, MicroscopyImage
from .model import SegmentationModel
from .pipeline import Pipeline
from .typing import ScalarArray, UInt16Array
from .utils import parallelized

CPU_COUNT = os.cpu_count()


@dataclass
class ImageBatchGenerator:
    """Generator for batches of microscopy images.

    This class creates batches of microscopy images for efficient processing by loading
    images from file paths only when needed. It supports optional shuffling of file paths
    before batch creation and provides a generator interface for iterating through batches.

    Attributes:
        paths:
            List of file paths to microscopy image files to be loaded and batched.
            Currently only supports ND2 files.
        sample_metadata_list:
            Optional list of metadata corresponding to each file microscopy image.
            If provided, must be the same length as file paths.
        batch_size: Number of images per batch. Default is 8.
        shuffle: Whether to shuffle file paths before batching. Default is False.
        random_state: Random seed for reproducible shuffling. Default is None.
    """

    paths: list[str | Path]
    sample_metadata_list: list[dict[str, Any]] | None = None
    batch_size: int = 8
    shuffle: bool = False
    random_state: int | None = None

    def __post_init__(self):
        """Validate and prepare file paths after initialization.

        Converts all file paths to Path objects, checks that each file exists, and verifies that
        only supported file formats (ND2) are used.

        Raises:
            FileNotFoundError: If any of the specified files don't exist.
            NotImplementedError: If any file has an unsupported format.
        """
        # Convert all file paths to Path
        self.paths = [Path(path) for path in self.paths]
        for path in self.paths:
            # Verify that all file paths exist
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} does not exist.")

            # Verify that file extension is supported
            if path.suffix != ".nd2":
                raise NotImplementedError("Only ND2 files are currently supported.")

        # Verify that sample metadata exists for each file path
        if self.sample_metadata_list is not None:
            if not len(self.paths) == len(self.sample_metadata_list):
                raise ValueError(
                    f"Length of sample_metadata_list ({len(self.sample_metadata_list)}) "
                    f"must match length of file paths ({len(self.paths)})."
                )
        else:
            # If no metadata provided, create empty dictionaries
            self.sample_metadata_list = [{} for _ in range(len(self.paths))]

    @property
    def num_batches(self) -> int:
        """Calculate the number of batches based on file path count and batch size."""
        if not self.paths:
            return 0
        _num_batches = len(self.paths) / self.batch_size
        return int(np.ceil(_num_batches))

    def generate_batches(self) -> Generator[ImageBatch, None, None]:
        """Generate batches of images for processing.

        If shuffle is True, creates a shuffled copy of the file paths list before generating
        batches. Images are loaded from disk only when a batch is being created, which minimizes
        memory usage.

        Yields:
            ImageBatch: Batches of microscopy images loaded from the specified file paths.
        """
        if not self.paths:
            return

        # Create copies to avoid modifying original data
        paths_to_process = self.paths.copy()
        metadata_to_process = self.sample_metadata_list.copy()

        # Shuffle both paths and metadata together if shuffle is enabled
        if self.shuffle:
            indices = list(range(len(paths_to_process)))
            shuffled_indices = sklearn.utils.shuffle(indices, random_state=self.random_state)

            # Apply the shuffled indices to both lists
            paths_to_process = [paths_to_process[i] for i in shuffled_indices]
            metadata_to_process = [metadata_to_process[i] for i in shuffled_indices]

        for i in range(0, len(paths_to_process), self.batch_size):
            batch_paths = paths_to_process[i : i + self.batch_size]
            batch_metadata = metadata_to_process[i : i + self.batch_size]
            batch_images = [
                MicroscopyImage.from_nd2_path(path, sample_metadata)
                for path, sample_metadata in zip(batch_paths, batch_metadata, strict=True)
            ]
            yield ImageBatch(batch_images)


@dataclass
class ImageBatch:
    """A batch of microscopy image data for parallelized processing.

    This class holds a collection of microscopy images and provides methods for
    batch processing with parallelization.

    Attributes:
        images:
            List of MicroscopyImage instances to process as a batch.
        num_workers:
            Number of threads to allocate for parallelization. Default is calculated based on
            batch size and CPU count.

    Properties:
        batch_size: Number of images in the batch.
        channels: List of all unique Channel instances within the batch.
    """

    images: list[MicroscopyImage]
    num_workers: int | None = None
    processed_intensities_dict: dict[Channel, ScalarArray] | None = field(default_factory=dict)
    segmentation_masks_dict: dict[Channel, ScalarArray] | None = field(default_factory=dict)

    def __post_init__(self):
        """Set default num_workers if not provided, otherwise validate the provided value."""
        num_cores = CPU_COUNT or 1
        max_reasonable_workers = num_cores * 10

        if self.num_workers is None:
            # Use default calculation when no explicit value provided
            self.num_workers = min(self.batch_size, max_reasonable_workers)
        else:
            # Cap explicit num_workers to reasonable bounds
            self.num_workers = min(self.num_workers, max_reasonable_workers, self.batch_size)

    def __len__(self) -> int:
        """Return the number of images in the batch."""
        return len(self.images)

    @property
    def batch_size(self) -> int:
        """Return the number of images in the batch."""
        return len(self)

    @property
    def channels(self) -> list[Channel]:
        """Return a sorted list of unique channels that appear in at least one of the images."""
        if not self.images:
            return []

        _channels = set(channel for image in self.images for channel in image.channels)
        return sorted(_channels, key=lambda c: c.value)

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        sample_metadata_list: list[dict[str, Any]] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> ImageBatch:
        """Create an ImageBatch directly from a list of file paths.

        This is a convenience method that creates and returns a single batch containing images
        loaded from the specified file paths. It internally uses ImageBatchGenerator with
        the provided parameters and returns the first batch.

        Args:
            paths: List of paths to ND2 files to load.
            sample_metadata_list: Optional list of metadata dictionaries corresponding to each file.
                If provided, must be the same length as paths.
            batch_size: Number of images per batch. If None, defaults to the number of file paths,
                loading all images into a single batch.
            shuffle: Whether to shuffle file paths before batching. Default is False.
            random_state: Random seed for reproducible shuffling. Default is None.

        Returns:
            ImageBatch: A batch containing images loaded from the specified file paths.

        Raises:
            FileNotFoundError: If any of the specified files don't exist.
            NotImplementedError: If any file has an unsupported format (only ND2 is supported).
            ValueError: If sample_metadata_list is provided but doesn't match length of file paths.
        """
        if batch_size is None:
            batch_size = len(paths)
        batch_generator = ImageBatchGenerator(
            paths, sample_metadata_list, batch_size, shuffle, random_state
        )
        return next(batch_generator.generate_batches())

    def get_intensities_from_channel(self, channel: Channel | str) -> UInt16Array:
        """Extract intensity data for a specific channel from all images in the batch.

        Collects the intensity arrays from the specified channel across all images in
        the batch and stacks them into a single batch array.

        Args:
            channel: The channel to extract, either as Channel enum or string name.

        Returns:
            intensities:
                A numpy array of intensity values with shape [batch_size, height, width],
                where the first dimension corresponds to the images in the batch.

        Raises:
            ValueError: If the specified channel is not present in all images in the batch.
        """
        intensities_list = []
        for image in self.images:
            intensities = image.get_intensities_from_channel(channel)
            intensities_list.append(intensities)
        return np.array(intensities_list)

    def apply_pipeline(
        self,
        pipeline: Pipeline,
        channel: Channel | str = Channel.BF,
        show_progress: bool = False,
    ) -> ImageBatch:
        """Apply an image processing pipeline to all images in the batch.

        Processes all images in parallel using the provided pipeline, applying it
        to the specified channel in each image.

        Args:
            pipeline: Pipeline object with image operations to apply.
            channel: Channel to process (default: brightfield).
            show_progress: Whether to show a progress bar during processing.

        Returns:
            self: This ImageBatch with processed images.
        """

        @parallelized(self.num_workers, show_progress)
        def apply_pipeline_to_image(intensities):
            return pipeline(intensities)

        # Apply to all images in parallel and store results
        batch_intensities = self.get_intensities_from_channel(channel)
        processed_batch_intensities = apply_pipeline_to_image(batch_intensities)
        self.processed_intensities_dict[channel] = np.array(processed_batch_intensities)
        return self

    def segment(
        self,
        model: SegmentationModel,
        channel: Channel | str = Channel.BF,
        cellpose_batch_size: int = 4,
        remove_edge_cells: bool = True,
        **cellpose_kwargs: dict[str, Any],
    ) -> ImageBatch:
        """Run cell segmentation on all images in the batch using a segmentation model.

        Applies cell segmentation to processed intensity data for the specified channel.
        Requires that `apply_pipeline` has been called first to populate the
        processed_intensities_dict.

        Args:
            model: A SegmentationModel instance to use for cell segmentation.
            channel: The channel to segment, either as Channel enum or string name.
            cellpose_batch_size: Number of 256x256 patches to run simultaneously on the GPU (not
                to be confused with self.batch_size). Default is 8.
            **cellpose_kwargs:
                Additional keyword arguments to pass to the model's run method,
                such as min_size.

        Returns:
            self: This ImageBatch with segmentation results stored in segmentation_masks_dict.

        Raises:
            KeyError: If the specified channel hasn't been processed with apply_pipeline.
            RuntimeError: If the segmentation model fails to process the images.

        Notes:
            - Segmentation is performed on the processed intensities, not the raw data
            - Results are stored in the segmentation_masks_dict attribute indexed by channel
            - The returned masks are integer arrays where each cell has a unique ID > 0,
              and 0 represents the background

        Examples:
            >>> batch = ImageBatch.from_paths(["image1.nd2", "image2.nd2"])
            >>> pipeline = Pipeline([...])
            >>> model = SegmentationModel(...)
            >>> batch.apply_pipeline(pipeline, Channel.BF)
            >>> batch.segment(model, Channel.BF, flow_threshold=0.4)
            >>> masks = batch.segmentation_masks_dict[Channel.BF]
        """
        # Convert string channel to enum if needed
        if isinstance(channel, str):
            channel = Channel[channel]

        # Get processed intensities as a list to pass to SegmentationModel
        batch_intensities = [frame for frame in self.processed_intensities_dict[channel]]

        # Run Cellpose
        masks = model.run(
            batch_intensities,
            batch_size=cellpose_batch_size,
            **cellpose_kwargs,
        )

        # Create segmentation masks
        segmentation_masks = []
        for mask_image, intensity_image in zip(masks, batch_intensities, strict=True):
            segmentation_mask = SegmentationMask(
                mask_image,
                intensity_image,
                remove_edge_cells,
            )
            segmentation_masks.append(segmentation_mask)

        self.segmentation_masks_dict[channel] = segmentation_masks
        return self
