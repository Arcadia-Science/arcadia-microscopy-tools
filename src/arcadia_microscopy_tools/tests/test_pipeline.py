import numpy as np
import pytest

from arcadia_microscopy_tools.pipeline import ImageOperation, Pipeline, PipelineParallelized


# Simple test operations for testing
def double_intensity(intensities):
    """Double all intensity values."""
    return intensities * 2


def add_ten(intensities):
    """Add 10 to all intensity values."""
    return intensities + 10


def to_float_normalized(intensities):
    """Convert to float and normalize to [0, 1]."""
    return intensities.astype(float) / intensities.max()


def square_values(intensities):
    """Square all values."""
    return intensities**2


class TestImageOperation:
    def test_create_operation_no_args(self):
        op = ImageOperation(double_intensity)
        assert op.method == double_intensity
        assert op.args == ()
        assert op.kwargs == {}

    def test_create_operation_with_args(self):
        op = ImageOperation(np.add, 5)
        assert op.method == np.add
        assert op.args == (5,)

    def test_create_operation_with_kwargs(self):
        op = ImageOperation(np.clip, a_min=0, a_max=100)
        assert op.kwargs == {"a_min": 0, "a_max": 100}

    def test_call_operation(self):
        op = ImageOperation(double_intensity)
        image = np.array([1, 2, 3])
        result = op(image)
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_call_operation_with_args(self):
        op = ImageOperation(np.add, 10)
        image = np.array([1, 2, 3])
        result = op(image)
        np.testing.assert_array_equal(result, [11, 12, 13])

    def test_repr(self):
        op = ImageOperation(double_intensity)
        assert "double_intensity" in repr(op)


class TestPipeline:
    def test_create_pipeline(self):
        ops = [ImageOperation(double_intensity), ImageOperation(add_ten)]
        pipeline = Pipeline(operations=ops)
        assert len(pipeline) == 2
        assert pipeline.copy is False
        assert pipeline.preserve_dtype is True

    def test_create_pipeline_with_copy(self):
        ops = [ImageOperation(double_intensity)]
        pipeline = Pipeline(operations=ops, copy=True)
        assert pipeline.copy is True

    def test_create_pipeline_with_preserve_dtype_false(self):
        ops = [ImageOperation(to_float_normalized)]
        pipeline = Pipeline(operations=ops, preserve_dtype=False)
        assert pipeline.preserve_dtype is False

    def test_pipeline_requires_operations(self):
        with pytest.raises(ValueError, match="at least one operation"):
            Pipeline(operations=[])

    def test_pipeline_single_operation(self):
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)])
        image = np.array([1, 2, 3], dtype=np.uint16)
        result = pipeline(image)
        np.testing.assert_array_equal(result, [2, 4, 6])
        assert result.dtype == np.uint16

    def test_pipeline_multiple_operations(self):
        pipeline = Pipeline(operations=[ImageOperation(double_intensity), ImageOperation(add_ten)])
        image = np.array([1, 2, 3], dtype=np.uint16)
        result = pipeline(image)
        # First double: [2, 4, 6], then add 10: [12, 14, 16]
        np.testing.assert_array_equal(result, [12, 14, 16])
        assert result.dtype == np.uint16

    def test_pipeline_preserve_dtype_default(self):
        """Test that dtype is preserved by default when it changes."""
        pipeline = Pipeline(operations=[ImageOperation(to_float_normalized)])
        image = np.array([10, 20, 30], dtype=np.uint16)
        result = pipeline(image)
        # to_float_normalized returns float, but preserve_dtype=True should cast back
        assert result.dtype == np.uint16

    def test_pipeline_preserve_dtype_false(self):
        """Test that dtype can change when preserve_dtype=False."""
        pipeline = Pipeline(operations=[ImageOperation(to_float_normalized)], preserve_dtype=False)
        image = np.array([10, 20, 30], dtype=np.uint16)
        result = pipeline(image)
        # Should return float
        assert result.dtype in (np.float32, np.float64)
        np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1.0])

    def test_pipeline_with_2d_image(self):
        """Test pipeline with 2D image arrays."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)])
        image = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = pipeline(image)
        expected = np.array([[2, 4], [6, 8]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)


class TestPipelineParallelized:
    def test_create_pipeline_parallelized(self):
        ops = [ImageOperation(double_intensity)]
        pipeline = PipelineParallelized(operations=ops)
        assert len(pipeline) == 1
        assert pipeline.max_workers is None
        assert pipeline.copy is False
        assert pipeline.preserve_dtype is True

    def test_create_pipeline_with_max_workers(self):
        ops = [ImageOperation(double_intensity)]
        pipeline = PipelineParallelized(operations=ops, max_workers=4)
        assert pipeline.max_workers == 4

    def test_pipeline_parallelized_requires_operations(self):
        with pytest.raises(ValueError, match="at least one operation"):
            PipelineParallelized(operations=[])

    def test_pipeline_parallelized_3d_array(self):
        """Test parallel processing of 3D array (e.g., time series)."""
        pipeline = PipelineParallelized(operations=[ImageOperation(double_intensity)])
        # Create 3D array: (time, height, width)
        image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint16

    def test_pipeline_parallelized_preserve_dtype_default(self):
        """Test that dtype is preserved by default."""
        pipeline = PipelineParallelized(operations=[ImageOperation(to_float_normalized)])
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
        result = pipeline(image)
        # Should preserve uint16 dtype by default
        assert result.dtype == np.uint16

    def test_pipeline_parallelized_preserve_dtype_false(self):
        """Test that dtype can change when preserve_dtype=False."""
        pipeline = PipelineParallelized(
            operations=[ImageOperation(to_float_normalized)], preserve_dtype=False
        )
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
        result = pipeline(image)
        # Should return float
        assert result.dtype in (np.float32, np.float64)

    def test_pipeline_parallelized_multiple_operations(self):
        """Test multiple operations in parallel pipeline."""
        pipeline = PipelineParallelized(
            operations=[ImageOperation(double_intensity), ImageOperation(add_ten)]
        )
        image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint16)
        result = pipeline(image)
        # First double, then add 10
        expected = (image * 2) + 10
        np.testing.assert_array_equal(result, expected)

    def test_pipeline_parallelized_single_frame(self):
        """Test with single frame (edge case)."""
        pipeline = PipelineParallelized(operations=[ImageOperation(double_intensity)])
        image = np.array([[[1, 2], [3, 4]]], dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)

    def test_pipeline_parallelized_many_frames(self):
        """Test with many frames to ensure parallelization works."""
        pipeline = PipelineParallelized(
            operations=[ImageOperation(double_intensity)], max_workers=2
        )
        # Create 10 frames
        image = np.random.randint(0, 100, size=(10, 32, 32), dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)


class TestPipelineIntegration:
    """Integration tests for realistic use cases."""

    def test_normalization_workflow_preserve_dtype_false(self):
        """Test a realistic normalization workflow for ML preprocessing."""
        from arcadia_microscopy_tools.operations import rescale_by_percentile

        # Simulate 16-bit microscopy images (3 frames)
        image = np.random.randint(0, 65535, size=(3, 128, 128), dtype=np.uint16)

        pipeline = PipelineParallelized(
            operations=[
                ImageOperation(
                    rescale_by_percentile,
                    percentile_range=(2, 98),
                    out_range=(0, 1),
                )
            ],
            preserve_dtype=False,
        )

        result = pipeline(image)

        # Should be normalized to [0, 1] float range
        assert result.dtype in (np.float32, np.float64)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_normalization_workflow_preserve_dtype_true(self):
        """Test normalization with dtype preservation (legacy behavior)."""
        from arcadia_microscopy_tools.operations import rescale_by_percentile

        # Simulate 16-bit microscopy images
        image = np.random.randint(0, 65535, size=(3, 128, 128), dtype=np.uint16)

        pipeline = PipelineParallelized(
            operations=[
                ImageOperation(
                    rescale_by_percentile,
                    percentile_range=(2, 98),
                    out_range=(0, 65535),
                )
            ],
            preserve_dtype=True,
        )

        result = pipeline(image)

        # Should stay as uint16
        assert result.dtype == np.uint16

    def test_background_subtraction_and_normalization(self):
        """Test combining background subtraction with normalization."""
        from arcadia_microscopy_tools.operations import (
            rescale_by_percentile,
            subtract_background_dog,
        )

        # Create test image with background
        image = np.random.randint(100, 200, size=(2, 64, 64), dtype=np.uint16)

        pipeline = PipelineParallelized(
            operations=[
                ImageOperation(subtract_background_dog, low_sigma=1, high_sigma=10),
                ImageOperation(
                    rescale_by_percentile,
                    percentile_range=(1, 99),
                    out_range=(0, 1),
                ),
            ],
            preserve_dtype=False,
        )

        result = pipeline(image)

        # Should be float after processing
        assert result.dtype in (np.float32, np.float64)
        assert result.shape == image.shape
