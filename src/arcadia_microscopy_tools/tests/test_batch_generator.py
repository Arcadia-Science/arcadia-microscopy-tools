import pytest

from arcadia_microscopy_tools.batch import ImageBatchGenerator


def test_empty_batch_generator():
    """Test behavior with an empty list of file paths."""
    batch_generator = ImageBatchGenerator([])
    assert batch_generator.num_batches == 0
    batches = list(batch_generator.generate_batches())
    assert len(batches) == 0


def test_batch_generator_properties(path_list):
    """Test basic properties of the batch generator."""
    # Test with default batch size
    batch_generator = ImageBatchGenerator(path_list)
    assert batch_generator.batch_size == 8
    assert batch_generator.num_batches == 1

    # Test with custom batch size smaller than file path list
    batch_generator = ImageBatchGenerator(path_list, batch_size=2)
    assert batch_generator.batch_size == 2
    assert batch_generator.num_batches == 3


def test_batch_generator_with_shuffle(path_list):
    """Test that shuffle does not modify the original list of file paths."""
    # Use fixed random state for reproducibility
    batch_generator = ImageBatchGenerator(
        path_list, batch_size=3, shuffle=True, random_state=42
    )

    # Store original paths to verify they don't change
    original_paths = path_list.copy()

    # Generate batches
    batches = list(batch_generator.generate_batches())

    # Check that we get the expected number of batches
    assert len(batches) == 2
    assert len(batches[0].images) == 3
    assert len(batches[1].images) == 2

    # Check that original list is not modified
    for i, path in enumerate(original_paths):
        assert path_list[i] == path


def test_sample_metadata_list(path_list):
    """Test that sample metadata is correctly associated with images."""
    # Create test metadata for each path
    sample_metadata_list = [
        {"sample_id": f"sample_{i}", "condition": f"condition_{i}"}
        for i in range(len(path_list))
    ]

    # Create batch generator with sample metadata
    batch_generator = ImageBatchGenerator(
        path_list,
        sample_metadata_list,
        batch_size=3,
    )

    # Check that each image has the correct metadata
    image_count = 0
    for batch in batch_generator.generate_batches():
        for image in batch.images:
            # Check that the metadata matches
            assert image.metadata.sample["sample_id"] == f"sample_{image_count}"
            assert image.metadata.sample["condition"] == f"condition_{image_count}"
            image_count += 1

    # Make sure we processed all images
    assert image_count == len(path_list)


def test_sample_metadata_validation(valid_multichannel_nd2_path):
    """Test that sample metadata validation works correctly."""
    # Create a test path list
    path_list = [valid_multichannel_nd2_path]

    # Test with mismatched metadata length
    with pytest.raises(ValueError):
        ImageBatchGenerator(
            path_list,
            sample_metadata_list=[{"a": 43}, {"b": 854}],
        )

    # Test with empty metadata list (should fail if validation requires matching length)
    with pytest.raises(ValueError):
        ImageBatchGenerator(path_list, sample_metadata_list=[])

    # Test with None metadata list
    # This should create a default empty metadata dict for each path
    batch_generator = ImageBatchGenerator(path_list, sample_metadata_list=None)
    assert len(batch_generator.sample_metadata_list) == len(path_list)
    assert all(isinstance(m, dict) for m in batch_generator.sample_metadata_list)
