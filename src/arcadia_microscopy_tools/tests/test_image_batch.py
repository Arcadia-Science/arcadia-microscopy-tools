import os
from pathlib import Path

import pytest

from arcadia_microscopy_tools.batch import ImageBatch

CPU_COUNT = os.cpu_count()


def test_image_batch_empty():
    """Test ImageBatch with an empty list of images."""
    batch = ImageBatch([])
    assert len(batch) == 0
    assert batch.batch_size == 0
    assert batch.channels == []
    assert batch.num_workers == 0


def test_image_batch_properties(image_list):
    """Test basic properties of ImageBatch.

    Notes:
        - Not ideal, reimplements some of the logic in ImageBatch.__post_init__()
    """
    batch = ImageBatch(image_list)
    num_available_cpus = CPU_COUNT or 1
    num_available_workers = max(1, num_available_cpus - 1)

    # Test dimensions
    assert len(batch) == 5
    assert batch.batch_size == 5

    # Test worker count
    num_expected_workers = min(len(image_list), num_available_workers)
    assert batch.num_workers == num_expected_workers

    # Test excessive worker count (should be capped)
    excessive_batch = ImageBatch(image_list, num_workers=1000)
    assert excessive_batch.num_workers == num_expected_workers


def test_image_batch_channels(image_list, multichannel_image):
    """Test extraction of unique channels from images."""
    # Test with single image
    single_image_batch = ImageBatch([multichannel_image])
    expected_channels = multichannel_image.channels
    assert set(single_image_batch.channels) == set(expected_channels)

    # Test with multiple images
    multi_image_batch = ImageBatch(image_list)
    # All unique channels across all images should be included
    all_channels = set()
    for image in image_list:
        all_channels.update(image.channels)
    assert set(multi_image_batch.channels) == all_channels

    # Verify channels are sorted
    assert multi_image_batch.channels == sorted(all_channels, key=lambda c: c.value)


def test_from_paths(path_list):
    """Test the from_paths class method."""
    # Test with default settings
    batch = ImageBatch.from_paths(path_list)

    # Check that the batch contains all images from the paths
    assert len(batch) == len(path_list)

    # Test with custom batch_size
    batch_with_size = ImageBatch.from_paths(path_list, batch_size=2)
    assert len(batch_with_size) == 2


def test_from_paths_with_sample_metadata(path_list):
    """Test that from_paths works correctly with sample metadata."""
    # Create test metadata for each path
    sample_metadata_list = [
        {"filename": path.name, "sample_id": f"sample_{i}", "experiment": i}
        for i, path in enumerate(path_list)
    ]

    # Use from_paths with sample metadata
    batch = ImageBatch.from_paths(path_list, sample_metadata_list)

    # Check that each image has the correct metadata
    for i, image in enumerate(batch.images):
        # Check that the metadata matches what we expect for this path
        assert image.metadata.sample["sample_id"] == f"sample_{i}"
        assert image.metadata.sample["experiment"] == i


def test_from_paths_errors():
    """Test error handling in the from_paths class method."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        ImageBatch.from_paths([Path("/non/existent/file.nd2")])

    # Test with unsupported file extension
    with pytest.raises(NotImplementedError):
        ImageBatch.from_paths(
            [Path(__file__)]
        )  # use this test file as an example of a non-ND2 file
