from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from arcadia_microscopy_tools.model import SegmentationModel


class TestSegmentationModel:
    """Test suite for SegmentationModel class."""

    def test_default_initialization(self):
        """Test that model initializes with correct default values."""
        model = SegmentationModel()

        assert model.default_cell_diameter_px == 30
        assert model.default_flow_threshold == 0.4
        assert model.default_cellprob_threshold == 0
        assert model.default_num_iterations is None
        assert model.default_batch_size == 8
        assert model.device is not None  # Should be auto-assigned
        assert model._model is None

    def test_custom_initialization(self):
        """Test that model initializes with custom values."""
        device = torch.device("cpu")
        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.5,
            default_cellprob_threshold=-2.0,
            default_num_iterations=10,
            default_batch_size=16,
            device=device,
        )

        assert model.default_cell_diameter_px == 25
        assert model.default_flow_threshold == 0.5
        assert model.default_cellprob_threshold == -2.0
        assert model.default_num_iterations == 10
        assert model.default_batch_size == 16
        assert model.device == device

    def test_cell_diameter_validation(self):
        """Test validation of cell_diameter_px parameter."""
        image_data = np.random.rand(100, 100)

        model1 = SegmentationModel(default_cell_diameter_px=0)
        with pytest.raises(ValueError, match="Cell diameter.*"):
            model1.segment(image_data)

        model2 = SegmentationModel(default_cell_diameter_px=-5)
        with pytest.raises(ValueError, match="Cell diameter.*"):
            model2.segment(image_data)

    def test_flow_threshold_validation(self):
        """Test validation of flow_threshold parameter."""
        image_data = np.random.rand(100, 100)
        model = SegmentationModel(default_flow_threshold=-0.1)
        with pytest.raises(ValueError, match="Flow threshold.*"):
            model.segment(image_data)

        # Valid values should not raise
        SegmentationModel(default_flow_threshold=0.0)
        SegmentationModel(default_flow_threshold=1.0)

    def test_cellprob_threshold_validation(self):
        """Test validation of cellprob_threshold parameter."""
        model1 = SegmentationModel(default_cellprob_threshold=-11)
        image_data = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="Cell probability threshold.*"):
            model1.segment(image_data)

        model2 = SegmentationModel(default_cellprob_threshold=11)
        with pytest.raises(ValueError, match="Cell probability threshold.*"):
            model2.segment(image_data)

        # Valid values should not raise during initialization or validation
        SegmentationModel(default_cellprob_threshold=-10)
        SegmentationModel(default_cellprob_threshold=10)
        SegmentationModel(default_cellprob_threshold=0)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_find_best_device_cuda(self, mock_props, mock_name, mock_cuda):
        """Test device selection when CUDA is available."""
        mock_cuda.return_value = True
        mock_name.return_value = "NVIDIA RTX 3080"
        mock_props.return_value = Mock(total_memory=10737418240)  # 10GB

        model = SegmentationModel()
        device = model.find_best_available_device()

        assert device.type == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_find_best_device_mps(self, mock_mps, mock_cuda):
        """Test device selection when MPS is available but CUDA is not."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        model = SegmentationModel()
        device = model.find_best_available_device()

        assert device.type == "mps"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    @patch("torch.get_num_threads")
    def test_find_best_device_cpu(self, mock_threads, mock_mps, mock_cuda):
        """Test device selection when only CPU is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_threads.return_value = 8

        model = SegmentationModel()
        device = model.find_best_available_device()

        assert device.type == "cpu"

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_cellpose_model_caching(self, mock_cellpose_class):
        """Test that cellpose model is cached after first access."""
        mock_model = Mock()
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel()

        # First access should create model
        result1 = model.cellpose_model
        assert result1 == mock_model
        assert mock_cellpose_class.call_count == 1

        # Second access should return cached model
        result2 = model.cellpose_model
        assert result2 == mock_model
        assert mock_cellpose_class.call_count == 1  # Still only called once

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_cellpose_model_error_handling(self, mock_cellpose_class):
        """Test error handling in cellpose model loading."""
        mock_cellpose_class.side_effect = Exception("Model loading failed")

        model = SegmentationModel()

        with pytest.raises(RuntimeError):
            _ = model.cellpose_model

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_segment_method(self, mock_cellpose_class):
        """Test the segment method with mocked Cellpose model."""
        # Setup mock
        mock_model = Mock()
        mock_masks = np.array([[1, 2], [3, 0]], dtype=np.uint16)
        mock_flows = np.zeros((2, 2, 2))
        mock_styles = np.zeros((256,))
        mock_imgs = np.zeros((2, 2))

        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_imgs)
        mock_cellpose_class.return_value = mock_model

        # Create model and test data
        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.3,
            default_cellprob_threshold=-1,
            default_batch_size=16,
        )
        image_data = np.random.rand(256, 256)

        # Run segmentation
        result = model.segment(image_data, min_size=100)

        # Verify results
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, mock_masks.astype(np.int64))

        # Verify Cellpose was called with correct parameters
        mock_model.eval.assert_called_once_with(
            x=image_data,
            diameter=25,
            flow_threshold=0.3,
            cellprob_threshold=-1,
            niter=None,
            batch_size=16,
            min_size=100,
        )

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_segment_method_error_handling(self, mock_cellpose_class):
        """Test error handling in segment method."""
        mock_model = Mock()
        mock_model.eval.side_effect = Exception("Segmentation failed")
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel()
        image_data = np.random.rand(100, 100)

        with pytest.raises(RuntimeError):
            model.segment(image_data)

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_segment_with_iterations(self, mock_cellpose_class):
        """Test segment method with num_iterations parameter."""
        mock_model = Mock()
        mock_masks = np.array([[1, 0], [0, 2]], dtype=np.uint16)
        mock_model.eval.return_value = (mock_masks, None, None, None)
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel(default_num_iterations=5)
        image_data = np.random.rand(50, 50)

        result = model.segment(image_data)

        # Verify niter parameter was passed correctly
        call_args = mock_model.eval.call_args
        assert call_args.kwargs["niter"] == 5
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_segment_with_override_parameters(self, mock_cellpose_class):
        """Test segment method with parameter overrides."""
        mock_model = Mock()
        mock_masks = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        mock_model.eval.return_value = (mock_masks, None, None, None)
        mock_cellpose_class.return_value = mock_model

        # Create model with defaults
        model = SegmentationModel(
            default_cell_diameter_px=30,
            default_flow_threshold=0.4,
            default_cellprob_threshold=0,
            default_num_iterations=None,
            default_batch_size=8,
        )
        image_data = np.random.rand(100, 100)

        # Call segment with overrides
        result = model.segment(
            image_data,
            cell_diameter_px=50,
            flow_threshold=0.6,
            cellprob_threshold=-2,
            num_iterations=10,
            batch_size=16,
        )

        # Verify overrides were used, not defaults
        call_args = mock_model.eval.call_args
        assert call_args.kwargs["diameter"] == 50
        assert call_args.kwargs["flow_threshold"] == 0.6
        assert call_args.kwargs["cellprob_threshold"] == -2
        assert call_args.kwargs["niter"] == 10
        assert call_args.kwargs["batch_size"] == 16
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_batch_segment_method(self, mock_cellpose_class):
        """Test the batch_segment method with multiple images."""
        mock_model = Mock()
        mock_masks1 = np.array([[1, 2], [3, 0]], dtype=np.uint16)
        mock_masks2 = np.array([[0, 1], [2, 3]], dtype=np.uint16)

        # Mock to return different masks for each call
        mock_model.eval.side_effect = [
            (mock_masks1, None, None, None),
            (mock_masks2, None, None, None),
        ]
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.3,
            default_cellprob_threshold=-1,
        )
        batch_data = [np.random.rand(256, 256), np.random.rand(256, 256)]

        # Run batch segmentation
        results = model.batch_segment(batch_data)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.dtype == np.int64 for r in results if r is not None)
        if results[0] is not None:
            np.testing.assert_array_equal(results[0], mock_masks1.astype(np.int64))
        if results[1] is not None:
            np.testing.assert_array_equal(results[1], mock_masks2.astype(np.int64))

        # Verify Cellpose was called twice with correct parameters
        assert mock_model.eval.call_count == 2

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_batch_segment_with_failures(self, mock_cellpose_class):
        """Test batch_segment continues processing after failures."""
        mock_model = Mock()
        mock_masks1 = np.array([[1, 2], [3, 0]], dtype=np.uint16)
        mock_masks3 = np.array([[5, 6], [7, 8]], dtype=np.uint16)

        # Mock to succeed, fail, then succeed
        mock_model.eval.side_effect = [
            (mock_masks1, None, None, None),
            Exception("Segmentation failed"),
            (mock_masks3, None, None, None),
        ]
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel()
        batch_data = [
            np.random.rand(100, 100),
            np.random.rand(100, 100),
            np.random.rand(100, 100),
        ]

        # Run batch segmentation
        results = model.batch_segment(batch_data)

        # Verify results - failed image should be None
        assert len(results) == 3
        assert isinstance(results[0], np.ndarray)
        assert results[1] is None  # Failed segmentation
        assert isinstance(results[2], np.ndarray)
        np.testing.assert_array_equal(results[0], mock_masks1.astype(np.int64))
        np.testing.assert_array_equal(results[2], mock_masks3.astype(np.int64))

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_batch_segment_with_overrides(self, mock_cellpose_class):
        """Test batch_segment with parameter overrides."""
        mock_model = Mock()
        mock_masks = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        mock_model.eval.return_value = (mock_masks, None, None, None)
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel(
            default_cell_diameter_px=30,
            default_flow_threshold=0.4,
        )
        batch_data = [np.random.rand(100, 100), np.random.rand(100, 100)]

        # Call with overrides
        results = model.batch_segment(
            batch_data,
            cell_diameter_px=50,
            flow_threshold=0.6,
            batch_size=16,
        )

        # Verify all calls used the overridden parameters
        assert len(results) == 2
        assert mock_model.eval.call_count == 2
        for call in mock_model.eval.call_args_list:
            assert call.kwargs["diameter"] == 50
            assert call.kwargs["flow_threshold"] == 0.6
            assert call.kwargs["batch_size"] == 16

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_batch_segment_validation_error(self, mock_cellpose_class):
        """Test batch_segment raises validation errors before processing."""
        mock_model = Mock()
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel()
        batch_data = [np.random.rand(100, 100)]

        # Invalid parameter should raise ValueError before any processing
        with pytest.raises(ValueError, match="Cell diameter.*"):
            model.batch_segment(batch_data, cell_diameter_px=-5)

        # Verify no images were processed
        mock_model.eval.assert_not_called()

    def test_resolve_and_validate_parameters_with_defaults(self):
        """Test parameter resolution uses defaults when no overrides provided."""
        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.3,
            default_cellprob_threshold=-1,
            default_num_iterations=5,
            default_batch_size=16,
        )

        params = model._resolve_and_validate_parameters(
            cell_diameter_px=None,
            flow_threshold=None,
            cellprob_threshold=None,
            num_iterations=None,
            batch_size=None,
        )

        assert params["diameter"] == 25
        assert params["flow_threshold"] == 0.3
        assert params["cellprob_threshold"] == -1
        assert params["niter"] == 5
        assert params["batch_size"] == 16

    def test_resolve_and_validate_parameters_with_overrides(self):
        """Test parameter resolution uses overrides when provided."""
        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.3,
            default_cellprob_threshold=-1,
            default_num_iterations=5,
            default_batch_size=16,
        )

        params = model._resolve_and_validate_parameters(
            cell_diameter_px=50,
            flow_threshold=0.6,
            cellprob_threshold=2,
            num_iterations=10,
            batch_size=32,
        )

        assert params["diameter"] == 50
        assert params["flow_threshold"] == 0.6
        assert params["cellprob_threshold"] == 2
        assert params["niter"] == 10
        assert params["batch_size"] == 32

    def test_resolve_and_validate_parameters_mixed(self):
        """Test parameter resolution with mix of defaults and overrides."""
        model = SegmentationModel(
            default_cell_diameter_px=25,
            default_flow_threshold=0.3,
            default_cellprob_threshold=-1,
        )

        params = model._resolve_and_validate_parameters(
            cell_diameter_px=50,  # Override
            flow_threshold=None,  # Use default
            cellprob_threshold=2,  # Override
            num_iterations=10,  # Override (default was None)
            batch_size=None,  # Use default
        )

        assert params["diameter"] == 50  # Overridden
        assert params["flow_threshold"] == 0.3  # Default
        assert params["cellprob_threshold"] == 2  # Overridden
        assert params["niter"] == 10  # Overridden
        assert params["batch_size"] == 8  # Default

    def test_resolve_and_validate_parameters_validation_errors(self):
        """Test parameter validation in _resolve_and_validate_parameters."""
        model = SegmentationModel()

        # Test invalid diameter
        with pytest.raises(ValueError, match="Cell diameter.*positive"):
            model._resolve_and_validate_parameters(
                cell_diameter_px=-5,
                flow_threshold=None,
                cellprob_threshold=None,
                num_iterations=None,
                batch_size=None,
            )

        # Test invalid flow_threshold
        with pytest.raises(ValueError, match="Flow threshold.*non-negative"):
            model._resolve_and_validate_parameters(
                cell_diameter_px=None,
                flow_threshold=-0.5,
                cellprob_threshold=None,
                num_iterations=None,
                batch_size=None,
            )

        # Test cellprob_threshold too low
        with pytest.raises(ValueError, match="Cell probability threshold.*between -10 and 10"):
            model._resolve_and_validate_parameters(
                cell_diameter_px=None,
                flow_threshold=None,
                cellprob_threshold=-15,
                num_iterations=None,
                batch_size=None,
            )

        # Test cellprob_threshold too high
        with pytest.raises(ValueError, match="Cell probability threshold.*between -10 and 10"):
            model._resolve_and_validate_parameters(
                cell_diameter_px=None,
                flow_threshold=None,
                cellprob_threshold=15,
                num_iterations=None,
                batch_size=None,
            )

    def test_resolve_and_validate_parameters_edge_cases(self):
        """Test parameter validation at boundary values."""
        model = SegmentationModel()

        # Test minimum valid values
        params = model._resolve_and_validate_parameters(
            cell_diameter_px=0.1,  # Just above 0
            flow_threshold=0.0,  # Minimum valid
            cellprob_threshold=-10,  # Minimum valid
            num_iterations=None,
            batch_size=None,
        )
        assert params["diameter"] == 0.1
        assert params["flow_threshold"] == 0.0
        assert params["cellprob_threshold"] == -10

        # Test maximum valid values for cellprob_threshold
        params = model._resolve_and_validate_parameters(
            cell_diameter_px=None,
            flow_threshold=None,
            cellprob_threshold=10,  # Maximum valid
            num_iterations=None,
            batch_size=None,
        )
        assert params["cellprob_threshold"] == 10
