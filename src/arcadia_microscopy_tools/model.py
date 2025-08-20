import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from cellpose.models import CellposeModel

from .typing import FloatArray, IntArray

logger = logging.getLogger(__name__)


@dataclass
class SegmentationModel:
    """CellposeModel wrapper class to facilitate high-throughput cell segmentation.

    This class wraps the Cellpose-SAM deep learning model for cell segmentation,
    providing a simplified interface for batch processing of microscopy images.

    Attributes:
        cell_diameter_px: Expected cell diameter in pixels. Used as `diameter` parameter
            for Cellpose evaluation. Default is 30 pixels.
        flow_threshold: Flow error threshold for mask generation. Higher values result
            in fewer masks. Must be >= 0. Default is 0.4.
        cellprob_threshold: Cell probability threshold for mask generation. Higher values
            result in fewer and more confident masks. Must be between -10 and 10. Default is 0.
        num_iterations: Number of iterations for segmentation algorithm. If None, uses
            Cellpose default. Default is None.
        device: PyTorch device for model computation. If None, automatically selects
            the best available device (CUDA > MPS > CPU).

    Notes:
        - Channels are no longer an input in Cellpose v4.x (Cellpose-SAM). The model will use the
          first 3 channels of the input image and truncate the rest. It is also (allegedly)
          channel-invariant -- order of channels in the image shouldn't matter.
        - Cellpose-SAM has been trained on images with ROI diameters from size 7.5 to 120, with a
          mean diameter of 30 pixels, such that it should be fairly robust to different cell sizes.

    See also:
        - https://cellpose.readthedocs.io/en/latest/settings.html#settings
    """

    cell_diameter_px: int = 30
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0
    num_iterations: int = None
    device: torch.device = field(default=None)
    _model: CellposeModel = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.cell_diameter_px <= 0:
            raise ValueError(f"Cell diameter [px] must be positive, got {self.cell_diameter_px}")

        if self.flow_threshold < 0:
            raise ValueError(f"Flow threshold must be non-negative, got {self.flow_threshold}")

        if not (-10 <= self.cellprob_threshold <= 10):
            raise ValueError(
                "Cell probability threshold must be between -10 and 10, "
                f"got {self.cellprob_threshold}"
            )

        if self.device is None:
            self.device = self.find_best_available_device()

    @property
    def cellpose_model(self) -> CellposeModel:
        """Lazy-load and cache the Cellpose model."""
        if self._model is None:
            logger.info(f"Loading Cellpose-SAM model on {self.device}")
            try:
                self._model = CellposeModel(device=self.device)
            except Exception as e:
                logger.error(f"Failed to load Cellpose model: {e}")
                raise RuntimeError from e
        return self._model

    def find_best_available_device(self) -> torch.device:
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

    def run(
        self,
        batch_intensities: list[FloatArray],
        batch_size: int = 8,
        **cellpose_kwargs: dict[str, Any],
    ) -> IntArray:
        """Run cell segmentation using Cellpose.

        Args:
            batch_intensities: Input list of image intensities with shape ([channel], height, width)
                where the channel dimension is optional. Intensity values should be normalized
                floats, typically in range [0, 1].
            batch_size: Number of 256x256 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Default is 8.
            **cellpose_kwargs: Additional keyword arguments passed to CellposeModel.eval().
                Common options include 'min_size' (minimum cell size in pixels).

        Returns:
            IntArray: Output batch of labeled masks with shape (batch, height, width).
                Each segmented cell has a unique positive integer ID, background is 0.

        Raises:
            RuntimeError: If the Cellpose model fails during segmentation.

        See also:
            - For full list of optional cellpose_kwargs, see:
              https://cellpose.readthedocs.io/en/latest/api.html#id0

        Notes:
            - Input for batch processing must be a list for Cellpose to recognize that the input is
              multiple images. Otherwise Cellpose will misinterpret the batch dimension as channels
              and truncate to the first 3.
            - Cellpose can scale well on CUDA GPUs with large batches, the benchmarks show speed
              improvements with batch sizes up to 32. But Apple's PyTorch MPS backend isn't as
              optimized for deep CNN inference throughput, so increasing batch size quickly hits
              bandwidth/kernel-scheduling limits and stops helping. This is a known theme in MPS
              discussions/benchmarks.
            - At the time of writing, Cellpose documentation for v4.x contains a fair amount of
              mistakes -- a lot of the docstrings haven't been updated since v3.
        """
        try:
            masks_uint16, *_ = self.cellpose_model.eval(
                x=batch_intensities,
                batch_size=batch_size,
                diameter=self.cell_diameter_px,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                niter=self.num_iterations,
                **cellpose_kwargs,
            )
        except Exception as e:
            logger.error(f"Cellpose segmentation failed: {e}")
            raise RuntimeError from e

        return np.array(masks_uint16, dtype=np.int64)
