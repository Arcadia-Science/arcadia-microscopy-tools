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

        assert model.cell_diameter_px == 30
        assert model.flow_threshold == 0.4
        assert model.cellprob_threshold == 0
        assert model.num_iterations is None
        assert model.device is not None  # Should be auto-assigned
        assert model._model is None

    def test_custom_initialization(self):
        """Test that model initializes with custom values."""
        device = torch.device("cpu")
        model = SegmentationModel(
            cell_diameter_px=25,
            flow_threshold=0.5,
            cellprob_threshold=-2.0,
            num_iterations=10,
            device=device,
        )

        assert model.cell_diameter_px == 25
        assert model.flow_threshold == 0.5
        assert model.cellprob_threshold == -2.0
        assert model.num_iterations == 10
        assert model.device == device

    def test_cell_diameter_validation(self):
        """Test validation of cell_diameter_px parameter."""
        with pytest.raises(ValueError, match="Cell diameter.*"):
            SegmentationModel(cell_diameter_px=0)

        with pytest.raises(ValueError, match="Cell diameter.*"):
            SegmentationModel(cell_diameter_px=-5)

    def test_flow_threshold_validation(self):
        """Test validation of flow_threshold parameter."""
        with pytest.raises(ValueError, match="Flow threshold.*"):
            SegmentationModel(flow_threshold=-0.1)

        # Valid values should not raise
        SegmentationModel(flow_threshold=0.0)
        SegmentationModel(flow_threshold=1.0)

    def test_cellprob_threshold_validation(self):
        """Test validation of cellprob_threshold parameter."""
        with pytest.raises(ValueError, match="Cell probability threshold.*"):
            SegmentationModel(cellprob_threshold=-11)

        with pytest.raises(ValueError, match="Cell probability threshold.*"):
            SegmentationModel(cellprob_threshold=11)

        # Valid values should not raise
        SegmentationModel(cellprob_threshold=-10)
        SegmentationModel(cellprob_threshold=10)
        SegmentationModel(cellprob_threshold=0)

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
    def test_run_method(self, mock_cellpose_class):
        """Test the run method with mocked Cellpose model."""
        # Setup mock
        mock_model = Mock()
        mock_masks = np.array([[[1, 2], [3, 0]], [[0, 1], [2, 3]]], dtype=np.uint16)
        mock_flows = np.zeros((2, 2, 2, 2))
        mock_styles = np.zeros((2, 256))
        mock_imgs = np.zeros((2, 2, 2))

        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_imgs)
        mock_cellpose_class.return_value = mock_model

        # Create model and test data
        model = SegmentationModel(cell_diameter_px=25, flow_threshold=0.3, cellprob_threshold=-1)
        batch_data = [np.random.rand(256, 256) for _ in range(2)]

        # Run segmentation
        result = model.run(batch_data, batch_size=8, min_size=100)

        # Verify results
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, mock_masks.astype(np.int64))

        # Verify Cellpose was called with correct parameters
        mock_model.eval.assert_called_once_with(
            x=batch_data,
            batch_size=8,
            diameter=25,
            flow_threshold=0.3,
            cellprob_threshold=-1,
            niter=None,
            min_size=100,
        )

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_run_method_error_handling(self, mock_cellpose_class):
        """Test error handling in run method."""
        mock_model = Mock()
        mock_model.eval.side_effect = Exception("Segmentation failed")
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel()
        batch_data = np.random.rand(1, 100, 100)

        with pytest.raises(RuntimeError):
            model.run([batch_data])

    @patch("arcadia_microscopy_tools.model.CellposeModel")
    def test_run_with_iterations(self, mock_cellpose_class):
        """Test run method with num_iterations parameter."""
        mock_model = Mock()
        mock_masks = np.array([[[1, 0], [0, 2]]], dtype=np.uint16)
        mock_model.eval.return_value = (mock_masks, None, None, None)
        mock_cellpose_class.return_value = mock_model

        model = SegmentationModel(num_iterations=5)
        batch_data = np.random.rand(1, 50, 50)

        model.run([batch_data])

        # Verify niter parameter was passed correctly
        call_args = mock_model.eval.call_args
        assert call_args.kwargs["niter"] == 5
