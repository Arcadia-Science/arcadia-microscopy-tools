from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import skimage as ski
from cellpose.utils import outlines_list

from .typing import IntArray, ScalarArray

OutlineExtractorMethod = Literal["cellpose", "skimage"]

CELL_PROPERTIES = [
    "label",
    "area",
    "area_convex",
    "perimeter",
    "eccentricity",
    "solidity",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
    "moments_hu",
    "inertia_tensor",
    "inertia_tensor_eigvals",
]


class CellposeOutlineExtractor:
    """Extract cell outlines using Cellpose's outlines_list function."""

    def extract_outlines(self, label_image: IntArray) -> list[ScalarArray]:
        """Extract outlines from label image."""
        return outlines_list(label_image, multiprocessing=False)


class SkimageOutlineExtractor:
    """Extract cell outlines using scikit-image's find_contours."""

    def extract_outlines(self, label_image: IntArray) -> list[ScalarArray]:
        """Extract outlines from label image."""
        # Get unique cell IDs (excluding background)
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]

        outlines = []
        for cell_id in unique_labels:
            cell_mask = (label_image == cell_id).astype(np.uint8)
            contours = ski.measure.find_contours(cell_mask, level=0.5)
            if contours:
                main_contour = max(contours, key=len)
                outlines.append(main_contour)
            else:
                outlines.append(np.array([]))
        return outlines


@dataclass
class MaskProcessor:
    """Process segmentation masks by removing edge cells and ensuring consecutive labels.

    Args:
        remove_edge_cells: Whether to remove cells touching image borders.
    """

    remove_edge_cells: bool = True

    def process_mask(self, mask_image: ScalarArray) -> IntArray:
        """Process a mask image by optionally removing edge cells and ensuring consecutive labels.

        Args:
            mask_image: Input mask array where each cell has a unique label.

        Returns:
            Processed label image with consecutive labels starting from 1.
        """
        _label_image = mask_image.copy()
        if self.remove_edge_cells:
            _label_image = ski.segmentation.clear_border(_label_image)

        # Ensure consecutive labels
        _label_image = ski.measure.label(_label_image)
        return _label_image.astype(np.int64)


@dataclass
class SegmentationMask:
    """Container for segmentation mask data and feature extraction.

    Args:
        mask_image: 2D integer array where each cell has a unique label (background=0).
        intensity_image: Optional intensity image for computing intensity-based features.
        remove_edge_cells: Whether to remove cells touching image borders.
        outline_extractor: Outline extraction method ("cellpose" or "skimage").
    """

    mask_image: ScalarArray
    intensity_image: ScalarArray | None = None
    remove_edge_cells: bool = True
    outline_extractor: OutlineExtractorMethod = "cellpose"

    def __post_init__(self):
        """Validate inputs and create processors."""
        # Validate mask_image
        if not isinstance(self.mask_image, np.ndarray):
            raise TypeError("mask_image must be a numpy array")
        if self.mask_image.ndim != 2:
            raise ValueError("mask_image must be a 2D array")
        if self.mask_image.min() < 0:
            raise ValueError("mask_image must have non-negative values")

        # Validate intensity_image if provided
        if (self.intensity_image is not None) and (
            self.mask_image.shape != self.intensity_image.shape
        ):
            raise ValueError("Intensity image must have same shape as mask image.")

        # Create mask processor
        self._mask_processor = MaskProcessor(remove_edge_cells=self.remove_edge_cells)

        # Create outline extractor
        if self.outline_extractor == "cellpose":
            self._outline_extractor = CellposeOutlineExtractor()
        else:  # Must be "skimage" due to Literal type
            self._outline_extractor = SkimageOutlineExtractor()

    @cached_property
    def label_image(self) -> IntArray:
        """Get processed label image with consecutive labels."""
        return self._mask_processor.process_mask(self.mask_image)

    @cached_property
    def num_cells(self) -> int:
        """Get the number of cells in the mask."""
        return int(self.label_image.max())

    @cached_property
    def cell_outlines(self) -> list[ScalarArray]:
        """Extract cell outlines using the configured outline extractor."""
        if self.num_cells == 0:
            return []

        return self._outline_extractor.extract_outlines(self.label_image)

    @cached_property
    def cell_properties(self) -> dict[str, ScalarArray]:
        """Extract cell properties using regionprops."""
        if self.num_cells == 0:
            return {prop: np.array([]) for prop in CELL_PROPERTIES}

        return ski.measure.regionprops_table(self.label_image, properties=CELL_PROPERTIES)
