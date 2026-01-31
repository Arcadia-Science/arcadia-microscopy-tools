import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import torch
from cellpose.models import CellposeModel

from .typing import Float64Array, Int64Array

logger = logging.getLogger(__name__)


class CellposeParams(TypedDict):
    """Parameters for CellposeModel.eval() with resolved types."""

    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    niter: int | None
    batch_size: int


@dataclass
class SegmentationModel:
    """CellposeModel wrapper class to facilitate high-throughput cell segmentation.

    This class wraps the Cellpose-SAM deep learning model for cell segmentation,
    providing a simplified interface for batch processing of microscopy images.

    Attributes:
        default_cell_diameter_px: Default expected cell diameter in pixels. Default is 30.
        default_flow_threshold: Default flow error threshold for mask generation. Higher values
            result in fewer masks. Must be >= 0. Default is 0.4.
        default_cellprob_threshold: Default cell probability threshold for mask generation.
            Higher values result in fewer and more confident masks. Must be between -10 and 10.
            Default is 0.
        default_num_iterations: Default number of iterations for segmentation algorithm.
            If None, uses Cellpose default (proportional to diameter).
        default_batch_size: Default number of 256x256 patches to run simultaneously on the GPU.
            Can be adjusted based on GPU memory. Default is 8.
        device: PyTorch device for model computation. If None, automatically selects
            the best available device (CUDA > MPS > CPU).

    Notes:
        - Cellpose-SAM uses the first 3 channels of input images and is channel-order invariant.
        - Trained on ROI diameters 7.5-120px (mean 30px). Specifying diameter is optional but
          can improve speed for large cells via downsampling (e.g., diameter=90 downsamples 3x).
        - Network outputs X/Y flows and cell probability (range ≈ -6 to +6). Pixels above
          cellprob_threshold are used for ROI detection. Decrease threshold for more ROIs,
          increase to reduce false positives from dim regions.
        - Flows simulate pixel dynamics over num_iterations iterations. Pixels converging to the
          same position form one ROI. Default num_iterations is proportional to diameter; longer
          ROIs may need more iterations (e.g., num_iterations=2000).
        - Cellpose can scale well on CUDA GPUs with large batches, the benchmarks show speed
          improvements with batch sizes up to 32. But Apple's PyTorch MPS backend isn't as
          optimized for deep CNN inference throughput, so increasing batch size quickly hits
          bandwidth/kernel-scheduling limits and stops helping. This is a known theme in MPS
          discussions/benchmarks.
        - See https://cellpose.readthedocs.io/en/latest/settings.html#settings for more details.
    """

    default_cell_diameter_px: float = 30
    default_flow_threshold: float = 0.4
    default_cellprob_threshold: float = 0
    default_num_iterations: int | None = None
    default_batch_size: int = 8
    device: torch.device | None = field(default=None)
    _model: CellposeModel | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Set device if not provided."""
        if self.device is None:
            self.device = self.find_best_available_device()

    def _resolve_and_validate_parameters(
        self,
        cell_diameter_px: float | None,
        flow_threshold: float | None,
        cellprob_threshold: float | None,
        num_iterations: int | None,
        batch_size: int | None,
    ) -> CellposeParams:
        """Resolve parameters from provided values or defaults, then validate.

        Args:
            cell_diameter_px: Expected cell diameter in pixels, or None for default.
            flow_threshold: Flow error threshold, or None for default.
            cellprob_threshold: Cell probability threshold, or None for default.
            num_iterations: Number of iterations, or None for default.
            batch_size: GPU batch size, or None for default.

        Returns:
            Dictionary with resolved and validated parameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        cellpose_params: CellposeParams = {
            "diameter": cell_diameter_px
            if cell_diameter_px is not None
            else self.default_cell_diameter_px,
            "flow_threshold": flow_threshold
            if flow_threshold is not None
            else self.default_flow_threshold,
            "cellprob_threshold": cellprob_threshold
            if cellprob_threshold is not None
            else self.default_cellprob_threshold,
            "niter": num_iterations if num_iterations is not None else self.default_num_iterations,
            "batch_size": batch_size if batch_size is not None else self.default_batch_size,
        }

        # Validate resolved parameters
        if cellpose_params["diameter"] <= 0:
            raise ValueError(
                f"Cell diameter [px] must be positive, got {cellpose_params['diameter']}"
            )
        if cellpose_params["flow_threshold"] < 0:
            raise ValueError(
                f"Flow threshold must be non-negative, got {cellpose_params['flow_threshold']}"
            )
        if not (-10 <= cellpose_params["cellprob_threshold"] <= 10):
            raise ValueError(
                "Cell probability threshold must be between -10 and 10, got "
                f"{cellpose_params['cellprob_threshold']}"
            )

        return cellpose_params

    @staticmethod
    def find_best_available_device() -> torch.device:
        """Get appropriate compute device (CUDA GPU, Apple Metal, or CPU).

        Determines the best available device for running the segmentation model:
            1. CUDA GPU if available (NVIDIA GPUs)
            2. MPS (Metal Performance Shaders) if available (Apple Silicon/AMD GPUs on macOS)
            3. CPU as fallback

        Returns:
            torch.device: The selected compute device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"Using CUDA GPU: {gpu_name} with {gpu_memory:.1f} GB memory")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS) for acceleration.")
        else:
            device = torch.device("cpu")
            cpu_count = torch.get_num_threads()
            logger.info(f"No GPU acceleration available. Using CPU with {cpu_count} threads.")
        return device

    @property
    def cellpose_model(self) -> CellposeModel:
        """Lazy-load and cache the Cellpose model."""
        if self._model is None:
            logger.info(f"Loading Cellpose-SAM model on {self.device}")
            try:
                self._model = CellposeModel(device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load Cellpose model: {e}") from e
        return self._model

    def segment(
        self,
        intensities: Float64Array,
        cell_diameter_px: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        **cellpose_kwargs: Any,
    ) -> Int64Array:
        """Run cell segmentation using Cellpose-SAM.

        Args:
            intensities: Input image intensities with shape ([channel], height, width).
                Intensity values should be normalized floats, typically in range [0, 1].
            cell_diameter_px: Expected cell diameter in pixels. If None, uses
                default_cell_diameter_px. See class attributes for details.
            flow_threshold: Flow error threshold. If None, uses default_flow_threshold.
                See class attributes for details.
            cellprob_threshold: Cell probability threshold. If None, uses
                default_cellprob_threshold. See class attributes for details.
            num_iterations: Number of iterations. If None, uses default_num_iterations.
                See class attributes for details.
            batch_size: GPU batch size. If None, uses default_batch_size.
                See class attributes for details.
            **cellpose_kwargs: Additional arguments passed to CellposeModel.eval().
                Full list: https://cellpose.readthedocs.io/en/latest/api.html#id0

        Returns:
            Segmentation mask as Int64Array.

        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If the Cellpose model fails during segmentation.
        """
        cellpose_params = self._resolve_and_validate_parameters(
            cell_diameter_px, flow_threshold, cellprob_threshold, num_iterations, batch_size
        )

        try:
            mask, *_ = self.cellpose_model.eval(x=intensities, **cellpose_params, **cellpose_kwargs)
        except Exception as e:
            raise RuntimeError(f"Cellpose segmentation failed: {e}") from e

        return mask.astype(np.int64)

    def batch_segment(
        self,
        intensities_batch: Sequence[Float64Array],
        cell_diameter_px: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        **cellpose_kwargs: Any,
    ) -> list[Int64Array | None]:
        """Run cell segmentation on multiple images using Cellpose-SAM.

        Args:
            intensities_batch: Sequence of input images, each with shape ([channel], height, width).
                Intensity values should be normalized floats, typically in range [0, 1].
            cell_diameter_px: Expected cell diameter. If None, uses default_cell_diameter_px.
                Applied to all images. See class attributes for details.
            flow_threshold: Flow error threshold. If None, uses default_flow_threshold.
                Applied to all images. See class attributes for details.
            cellprob_threshold: Cell probability threshold. If None, uses
                default_cellprob_threshold. Applied to all images. See class attributes for details.
            num_iterations: Number of iterations. If None, uses default_num_iterations.
                Applied to all images. See class attributes for details.
            batch_size: GPU batch size. If None, uses default_batch_size.
                See class attributes for details.
            **cellpose_kwargs: Additional arguments passed to CellposeModel.eval().
                Full list: https://cellpose.readthedocs.io/en/latest/api.html#id0

        Returns:
            List of segmentation mask arrays, one for each input image. Failed segmentations
            are represented as None in the output list, maintaining index alignment with the
            input list.

        Raises:
            ValueError: If parameters are out of valid ranges.

        Notes:
            All images are processed with the same parameters, which are resolved and
            validated once before processing. Each image is processed independently.
            If segmentation fails for an image, the error is logged and None is returned
            for that image, but processing continues for remaining images.
        """
        cellpose_params = self._resolve_and_validate_parameters(
            cell_diameter_px, flow_threshold, cellprob_threshold, num_iterations, batch_size
        )

        masks = []
        for i, intensities in enumerate(intensities_batch):
            try:
                mask, *_ = self.cellpose_model.eval(
                    x=intensities, **cellpose_params, **cellpose_kwargs
                )
                masks.append(mask.astype(np.int64))
            except Exception as e:
                logger.error(f"Cellpose segmentation failed on image {i}: {e}")
                masks.append(None)

        return masks
