import warnings

import numpy as np
import pytest

from arcadia_microscopy_tools.pipeline import ImageOperation, Pipeline


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
        assert op.func == double_intensity
        assert op.args == ()
        assert op.kwargs == {}

    def test_create_operation_with_args(self):
        op = ImageOperation(np.add, 5)
        assert op.func == np.add
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

    def test_frozen(self):
        op = ImageOperation(double_intensity)
        with pytest.raises(AttributeError):
            op.func = add_ten  # type: ignore[misc]

    def test_equality(self):
        op1 = ImageOperation(double_intensity)
        op2 = ImageOperation(double_intensity)
        assert op1 == op2

    def test_inequality(self):
        op1 = ImageOperation(double_intensity)
        op2 = ImageOperation(add_ten)
        assert op1 != op2

    def test_equality_with_args(self):
        op1 = ImageOperation(np.add, 5)
        op2 = ImageOperation(np.add, 5)
        assert op1 == op2

        op3 = ImageOperation(np.add, 10)
        assert op1 != op3


class TestPipeline:
    def test_create_pipeline(self):
        ops = [ImageOperation(double_intensity), ImageOperation(add_ten)]
        pipeline = Pipeline(operations=ops)
        assert len(pipeline) == 2
        assert pipeline.copy is False
        assert pipeline.preserve_dtype is False
        assert pipeline.parallel is False

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
        np.testing.assert_array_equal(result, [12, 14, 16])
        assert result.dtype == np.uint16

    def test_pipeline_preserve_dtype_default(self):
        """Test that dtype can change by default when operations produce a different dtype."""
        pipeline = Pipeline(operations=[ImageOperation(to_float_normalized)])
        image = np.array([10, 20, 30], dtype=np.uint16)
        result = pipeline(image)
        assert result.dtype in (np.float32, np.float64)
        np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1.0])

    def test_pipeline_preserve_dtype_true(self):
        """Test that dtype is preserved when preserve_dtype=True."""
        pipeline = Pipeline(operations=[ImageOperation(to_float_normalized)], preserve_dtype=True)
        image = np.array([10, 20, 30], dtype=np.uint16)
        result = pipeline(image)
        assert result.dtype == np.uint16

    def test_pipeline_with_2d_image(self):
        """Test pipeline with 2D image arrays."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)])
        image = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = pipeline(image)
        expected = np.array([[2, 4], [6, 8]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_operations_coerced_to_list(self):
        """Test that a tuple of operations is converted to a list."""
        ops = (ImageOperation(double_intensity),)
        pipeline = Pipeline(operations=ops)  # type: ignore[arg-type]
        assert isinstance(pipeline.operations, list)

    def test_max_workers_validation(self):
        """Test that max_workers must be at least 1."""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            Pipeline(operations=[ImageOperation(double_intensity)], max_workers=0)

    def test_max_workers_negative(self):
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            Pipeline(operations=[ImageOperation(double_intensity)], max_workers=-1)

    def test_non_callable_operation_raises(self):
        """Test that non-callable operations raise TypeError."""
        with pytest.raises(TypeError, match="All operations must be callable"):
            Pipeline(operations=("not_a_function",))  # type: ignore[arg-type]

    def test_mixed_callable_non_callable_raises(self):
        """Test that a mix of callable and non-callable operations raises TypeError."""
        with pytest.raises(TypeError, match="All operations must be callable"):
            Pipeline(operations=(ImageOperation(double_intensity), 42))  # type: ignore[arg-type]


class TestPipelineParallel:
    def test_copy_true_parallel_warns(self):
        """Test that copy=True with parallel=True emits a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Pipeline(operations=[ImageOperation(double_intensity)], parallel=True, copy=True)
            assert len(w) == 1
            assert "copy=True has no effect" in str(w[0].message)

    def test_create_parallel_pipeline(self):
        ops = [ImageOperation(double_intensity)]
        pipeline = Pipeline(operations=ops, parallel=True)
        assert len(pipeline) == 1
        assert pipeline.parallel is True
        assert pipeline.max_workers is None

    def test_create_parallel_pipeline_with_max_workers(self):
        ops = [ImageOperation(double_intensity)]
        pipeline = Pipeline(operations=ops, parallel=True, max_workers=4)
        assert pipeline.max_workers == 4

    def test_parallel_requires_operations(self):
        with pytest.raises(ValueError, match="at least one operation"):
            Pipeline(operations=[], parallel=True)

    def test_parallel_rejects_2d_input(self):
        """Test that parallel mode raises on 2D input."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)], parallel=True)
        image = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        with pytest.raises(ValueError, match="at least 3D input"):
            pipeline(image)

    def test_parallel_rejects_1d_input(self):
        """Test that parallel mode raises on 1D input."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)], parallel=True)
        image = np.array([1, 2, 3], dtype=np.uint16)
        with pytest.raises(ValueError, match="at least 3D input"):
            pipeline(image)

    def test_parallel_3d_array(self):
        """Test parallel processing of 3D array (e.g., time series)."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)], parallel=True)
        image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint16

    def test_parallel_preserve_dtype_default(self):
        """Test that dtype can change by default in parallel mode."""
        pipeline = Pipeline(operations=[ImageOperation(to_float_normalized)], parallel=True)
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
        result = pipeline(image)
        assert result.dtype in (np.float32, np.float64)

    def test_parallel_preserve_dtype_true(self):
        """Test that dtype is preserved when preserve_dtype=True in parallel mode."""
        pipeline = Pipeline(
            operations=[ImageOperation(to_float_normalized)], preserve_dtype=True, parallel=True
        )
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
        result = pipeline(image)
        assert result.dtype == np.uint16

    def test_parallel_multiple_operations(self):
        """Test multiple operations in parallel pipeline."""
        pipeline = Pipeline(
            operations=[ImageOperation(double_intensity), ImageOperation(add_ten)], parallel=True
        )
        image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint16)
        result = pipeline(image)
        expected = (image * 2) + 10
        np.testing.assert_array_equal(result, expected)

    def test_parallel_single_frame(self):
        """Test with single frame (edge case)."""
        pipeline = Pipeline(operations=[ImageOperation(double_intensity)], parallel=True)
        image = np.array([[[1, 2], [3, 4]]], dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)

    def test_parallel_many_frames(self):
        """Test with many frames to ensure parallelization works."""
        pipeline = Pipeline(
            operations=[ImageOperation(double_intensity)], parallel=True, max_workers=2
        )
        image = np.random.randint(0, 100, size=(10, 32, 32), dtype=np.uint16)
        result = pipeline(image)
        expected = image * 2
        np.testing.assert_array_equal(result, expected)


class TestPipelineIntegration:
    """Integration tests for realistic use cases."""

    def test_normalization_workflow_preserve_dtype_false(self):
        """Test a realistic normalization workflow for ML preprocessing."""
        from arcadia_microscopy_tools.operations import rescale_by_percentile

        image = np.random.randint(0, 65535, size=(3, 128, 128), dtype=np.uint16)

        pipeline = Pipeline(
            operations=[
                ImageOperation(rescale_by_percentile, percentile_range=(2, 98), out_range=(0, 1)),
            ],
            preserve_dtype=False,
            parallel=True,
        )

        result = pipeline(image)

        assert result.dtype in (np.float32, np.float64)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_normalization_workflow_preserve_dtype_true(self):
        """Test normalization with dtype preservation."""
        from arcadia_microscopy_tools.operations import rescale_by_percentile

        image = np.random.randint(0, 65535, size=(3, 128, 128), dtype=np.uint16)

        pipeline = Pipeline(
            operations=[
                ImageOperation(
                    rescale_by_percentile, percentile_range=(2, 98), out_range=(0, 65535)
                ),
            ],
            preserve_dtype=True,
            parallel=True,
        )

        result = pipeline(image)

        assert result.dtype == np.uint16

    def test_background_subtraction_and_normalization(self):
        """Test combining background subtraction with normalization."""
        from arcadia_microscopy_tools.operations import (
            rescale_by_percentile,
            subtract_background_dog,
        )

        image = np.random.randint(100, 200, size=(2, 64, 64), dtype=np.uint16)

        pipeline = Pipeline(
            operations=[
                ImageOperation(subtract_background_dog, low_sigma=1, high_sigma=10),
                ImageOperation(rescale_by_percentile, percentile_range=(1, 99), out_range=(0, 1)),
            ],
            preserve_dtype=False,
            parallel=True,
        )

        result = pipeline(image)

        assert result.dtype in (np.float32, np.float64)
        assert result.shape == image.shape
