import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from cellpose.models import CellposeModel

from .typing import FloatArray, Int64Array

logger = logging.getLogger(__name__)


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
        """Validate default parameters and set device."""
        self._validate_parameters(
            self.default_cell_diameter_px,
            self.default_flow_threshold,
            self.default_cellprob_threshold,
        )

        if self.device is None:
            self.device = self.find_best_available_device()

    def _resolve_parameters(
        self,
        cell_diameter_px: float | None,
        flow_threshold: float | None,
        cellprob_threshold: float | None,
        num_iterations: int | None,
        batch_size: int | None,
    ) -> tuple[float, float, float, int | None, int]:
        """Resolve parameters by using provided values or falling back to defaults.

        Args:
            cell_diameter_px: Expected cell diameter in pixels, or None to use default.
            flow_threshold: Flow error threshold, or None to use default.
            cellprob_threshold: Cell probability threshold, or None to use default.
            num_iterations: Number of iterations, or None to use default.
            batch_size: Batch size, or None to use default.

        Returns:
            Tuple of (cell_diameter_px, flow_threshold, cellprob_threshold, num_iterations, batch_size)
            with defaults applied where parameters were None.
        """
        resolved_cell_diameter_px = (
            cell_diameter_px if cell_diameter_px is not None else self.default_cell_diameter_px
        )
        resolved_flow_threshold = (
            flow_threshold if flow_threshold is not None else self.default_flow_threshold
        )
        resolved_cellprob_threshold = (
            cellprob_threshold
            if cellprob_threshold is not None
            else self.default_cellprob_threshold
        )
        resolved_num_iterations = (
            num_iterations if num_iterations is not None else self.default_num_iterations
        )
        resolved_batch_size = batch_size if batch_size is not None else self.default_batch_size

        return (
            resolved_cell_diameter_px,
            resolved_flow_threshold,
            resolved_cellprob_threshold,
            resolved_num_iterations,
            resolved_batch_size,
        )

    @staticmethod
    def _validate_parameters(
        cell_diameter_px: float,
        flow_threshold: float,
        cellprob_threshold: float,
    ) -> None:
        """Validate segmentation parameters.

        Args:
            cell_diameter_px: Expected cell diameter in pixels.
            flow_threshold: Flow error threshold for mask generation.
            cellprob_threshold: Cell probability threshold for mask generation.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if cell_diameter_px <= 0:
            raise ValueError(f"Cell diameter [px] must be positive, got {cell_diameter_px}")
        if flow_threshold < 0:
            raise ValueError(f"Flow threshold must be non-negative, got {flow_threshold}")
        if not (-10 <= cellprob_threshold <= 10):
            raise ValueError(
                f"Cell probability threshold must be between -10 and 10, got {cellprob_threshold}"
            )

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
        intensities: FloatArray,
        cell_diameter_px: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        **cellpose_kwargs: Any,
    ) -> Int64Array:
        """Run cell segmentation using Cellpose-SAM.

        Args:
            intensities: Input image intensities with shape ([channel], height, width)
                where the channel dimension is optional. Intensity values should be normalized
                floats, typically in range [0, 1].
            cell_diameter_px: Expected cell diameter in pixels. If None, uses the default
                value set during model initialization.
            flow_threshold: Flow error threshold for mask generation. Higher values result
                in fewer masks. Must be >= 0. If None, uses the default value.
            cellprob_threshold: Cell probability threshold for mask generation. Higher values
                result in fewer and more confident masks. Must be between -10 and 10.
                If None, uses the default value.
            num_iterations: Number of iterations for segmentation algorithm. If none, uses the
                default value (which may itself be None, triggering Cellpose's internal default).
            batch_size: Number of 256x256 patches to run simultaneously on the GPU.
                Can be adjusted based on GPU memory. If None, uses the default value.
            **cellpose_kwargs: Additional keyword arguments passed to CellposeModel.eval().
                Common options include 'min_size' (minimum cell size in pixels).

        Returns:
            SegmentationMask: Container with the segmentation mask and feature extraction methods.

        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If the Cellpose model fails during segmentation.

        See also:
            - For full list of optional cellpose_kwargs, see:
              https://cellpose.readthedocs.io/en/latest/api.html#id0
        """
        # Resolve and validate parameters
        (
            cell_diameter_px,
            flow_threshold,
            cellprob_threshold,
            num_iterations,
            batch_size,
        ) = self._resolve_parameters(
            cell_diameter_px, flow_threshold, cellprob_threshold, num_iterations, batch_size
        )
        self._validate_parameters(cell_diameter_px, flow_threshold, cellprob_threshold)

        try:
            masks_uint16, *_ = self.cellpose_model.eval(
                x=intensities,
                batch_size=batch_size,
                diameter=cell_diameter_px,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                niter=num_iterations,
                **cellpose_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Cellpose segmentation failed: {e}") from e

        return masks_uint16.astype(Int64Array)

    def batch_segment(
        self,
        intensities_list: Sequence[FloatArray],
        cell_diameter_px: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        **cellpose_kwargs: Any,
    ) -> list[Int64Array]:
        """Run cell segmentation on multiple images using Cellpose-SAM.

        Args:
            intensities_list: Sequence of input images, each with shape ([channel], height, width)
                where the channel dimension is optional. Intensity values should be normalized
                floats, typically in range [0, 1].
            cell_diameter_px: Expected cell diameter in pixels. If None, uses the default
                value set during model initialization. Applied to all images.
            flow_threshold: Flow error threshold for mask generation. Higher values result
                in fewer masks. Must be >= 0. If None, uses the default value. Applied to all images.
            cellprob_threshold: Cell probability threshold for mask generation. Higher values
                result in fewer and more confident masks. Must be between -10 and 10.
                If None, uses the default value. Applied to all images.
            num_iterations: Number of iterations for segmentation algorithm. If None, uses the
                default value (which may itself be None, triggering Cellpose's internal default).
                Applied to all images.
            batch_size: Number of 256x256 patches to run simultaneously on the GPU.
                Can be adjusted based on GPU memory. If None, uses the default value.
            **cellpose_kwargs: Additional keyword arguments passed to CellposeModel.eval().
                Common options include 'min_size' (minimum cell size in pixels).

        Returns:
            List of segmentation mask arrays, one for each input image.

        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If the Cellpose model fails during segmentation.

        Notes:
            - All images are processed with the same segmentation parameters.
            - Parameters are resolved and validated once before processing any images.
            - Each image is processed independently; failures on one image will halt processing.
        """
        # Resolve and validate parameters once for all images
        (
            cell_diameter_px,
            flow_threshold,
            cellprob_threshold,
            num_iterations,
            batch_size,
        ) = self._resolve_parameters(
            cell_diameter_px, flow_threshold, cellprob_threshold, num_iterations, batch_size
        )
        self._validate_parameters(cell_diameter_px, flow_threshold, cellprob_threshold)

        # Process each image
        results = []
        for i, intensities in enumerate(intensities_list):
            try:
                masks_uint16, *_ = self.cellpose_model.eval(
                    x=intensities,
                    batch_size=batch_size,
                    diameter=cell_diameter_px,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    niter=num_iterations,
                    **cellpose_kwargs,
                )
                results.append(masks_uint16.astype(Int64Array))
            except Exception as e:
                raise RuntimeError(f"Cellpose segmentation failed on image {i}: {e}") from e

        return results
