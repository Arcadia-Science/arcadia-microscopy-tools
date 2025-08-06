import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from cellpose.denoise import CellposeDenoiseModel

from .typing import FloatArray, IntArray, ScalarArray

logger = logging.getLogger(__name__)


@dataclass
class SegmentationModel:
    """Cell segmentation model based on Cellpose.

    This class wraps the Cellpose deep learning model for cell segmentation,
    providing a simplified interface for batch processing of microscopy images.

    Attributes:
        model_type: Cellpose model type (e.g. "cyto3", "nuclei").
        restore_type: Type of restoration/denoising to apply.
        batch_size: Number of images to process simultaneously.
        channels: List with two elements for cytoplasm and nucleus channels.
        diameter: Expected cell diameter in pixels.
        do_3D: Whether to use 3D segmentation.
    """

    model_type: str = "cyto3"
    restore_type: str = "denoise_cyto3"
    batch_size: int = 8
    channels: list[int] = field(default_factory=lambda: [0, 0])
    diameter: int = 10
    do_3D: bool = False
    _model: CellposeDenoiseModel | None = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not isinstance(self.channels, list) or len(self.channels) != 2:
            raise ValueError("channels must be a list with exactly 2 elements")

        if self.diameter <= 0:
            raise ValueError("diameter must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def device(self) -> torch.device:
        """Get appropriate compute device (CPU, CUDA GPU, or Apple Metal).

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
    def cellpose_model(self) -> CellposeDenoiseModel:
        """Lazy-load and cache the Cellpose model."""
        if self._model is None:
            logger.info(f"Loading {self.model_type} model with {self.restore_type} restoration.")
            try:
                self._model = CellposeDenoiseModel(
                    model_type=self.model_type,
                    restore_type=self.restore_type,
                    device=self.device,
                )
            except Exception as e:
                logger.error(f"Failed to load Cellpose model: {e}")
                raise

        return self._model

    def run_cellpose(
        self,
        array: FloatArray,
        return_all: bool = False,
        **kwargs,
    ) -> IntArray | tuple[IntArray, ScalarArray, ScalarArray, ScalarArray]:
        """Run cell segmentation using Cellpose.

        Args:
            array: Input image array with shape [batch, height, width].
            return_all: If True, return masks, flows, styles, and images; otherwise just masks.
            **kwargs: Additional keyword arguments passed to Cellpose model.eval().

        Returns:
            If return_all=False:
                IntArray: Labeled cell masks where each distinct cell has a unique integer ID
            If return_all=True:
                Tuple containing:
                - masks: Labeled cell masks
                - flows: Flow fields used for segmentation
                - styles: Style vectors
                - imgs: Processed input images

        Raises:
            RuntimeError: If the Cellpose model fails during segmentation
        """
        try:
            masks, flows, styles, imgs = self.cellpose_model.eval(
                x=array,
                batch_size=self.batch_size,
                channels=self.channels,
                diameter=self.diameter,
                do_3D=self.do_3D,
                **kwargs,
            )

            # Convert to int64 for consistency
            masks = masks.astype(np.int64)

            if return_all:
                return masks, flows, styles, imgs
            else:
                return masks

        except Exception as e:
            logger.error(f"Cellpose segmentation failed: {e}")
            raise RuntimeError(f"Cellpose segmentation failed: {e}") from e
