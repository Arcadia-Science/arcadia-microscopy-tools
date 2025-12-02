from __future__ import annotations
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import skimage as ski
from cellpose.utils import outlines_list

from .typing import BoolArray, FloatArray, Int64Array, ScalarArray

OutlineExtractorMethod = Literal["cellpose", "skimage"]

DEFAULT_CELL_PROPERTY_NAMES = [
    "label",
    "centroid",
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

    def extract_outlines(self, label_image: Int64Array) -> list[ScalarArray]:
        """Extract outlines from label image."""
        return outlines_list(label_image, multiprocessing=False)


class SkimageOutlineExtractor:
    """Extract cell outlines using scikit-image's find_contours."""

    def extract_outlines(self, label_image: Int64Array) -> list[ScalarArray]:
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

    def process_mask(self, mask_image: ScalarArray) -> Int64Array:
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
        property_names: List of property names to compute. If None, uses default property names.
    """

    mask_image: ScalarArray
    intensity_image: ScalarArray | None = None
    remove_edge_cells: bool = True
    outline_extractor: OutlineExtractorMethod = "cellpose"
    property_names: list[str] | None = None

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

        # Set default property names if none provided
        if self.property_names is None:
            self.property_names = DEFAULT_CELL_PROPERTY_NAMES.copy()

        # Create mask processor
        self._mask_processor = MaskProcessor(remove_edge_cells=self.remove_edge_cells)

        # Create outline extractor
        if self.outline_extractor == "cellpose":
            self._outline_extractor = CellposeOutlineExtractor()
        else:  # Must be "skimage" due to Literal type
            self._outline_extractor = SkimageOutlineExtractor()

    @cached_property
    def label_image(self) -> Int64Array:
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
        """Extract cell property values using regionprops."""
        if self.num_cells == 0:
            return {property_name: np.array([]) for property_name in self.property_names}

        return ski.measure.regionprops_table(
            self.label_image,
            properties=self.property_names,
            extra_properties=[circularity, volume],
        )

    @cached_property
    def centroids_yx(self) -> ScalarArray:
        """Get cell centroids as (y, x) coordinates.

        Extracts centroid coordinates from cell properties and returns them as a 2D array
        where each row represents one cell's centroid in (y, x) format.

        Returns:
            Array of shape (num_cells, 2) with centroid coordinates.
            Each row is [y_coordinate, x_coordinate] for one cell.
            Returns empty array if "centroid" is not included in property_names.

        Note:
            If "centroid" is not in property_names, issues a warning and returns an empty array.
        """
        if "centroid" not in self.property_names:
            warnings.warn(
                "Centroid property not available. Include 'centroid' in property_names "
                "to get centroid coordinates. Returning empty array.",
                UserWarning,
                stacklevel=2,
            )
            return np.array([]).reshape(0, 2)

        yc = self.cell_properties["centroid-0"]
        xc = self.cell_properties["centroid-1"]
        return np.array([yc, xc]).T


def circularity(
    region_mask: BoolArray,
    intensity_image: FloatArray | None = None,
) -> float:
    """Calculate the circularity of a cell region.

    Circularity is a shape metric that quantifies how close a shape is to a perfect circle.
    It is computed as (4π * area) / perimeter², ranging from 0 to 1, where 1 represents
    a perfect circle and lower values indicate more elongated or irregular shapes.

    Args:
        region_mask: Boolean mask of the cell region.
        intensity_image:
            Optional intensity image (unused but included for regionprops compatibility).

    Returns:
        Circularity value between 0 and 1. Returns 0 if perimeter is zero.
    """
    # regionprops expects a labeled image, so convert the mask (0/1)
    labeled_mask = region_mask.astype(np.int64, copy=False)

    # Compute standard region properties on this mask
    props = ski.measure.regionprops(labeled_mask)[0]
    area = float(props.area)
    perimeter = float(props.perimeter)

    if perimeter == 0.0:
        return 0.0

    return (4.0 * np.pi * area) / (perimeter**2)


def volume(
    region_mask: BoolArray,
    intensity_image: FloatArray | None = None,
) -> float:
    """Estimate the volume of a cell region.

    Volume is estimated by fitting an ellipse to the cell region and treating it as
    a prolate spheroid (ellipsoid of revolution). The ellipsoid is formed by rotating
    the fitted ellipse around its major axis, with volume = (4/3)π * a * b^2, where
    a is the semi-major axis and b is the semi-minor axis.

    Args:
        region_mask: Boolean mask of the cell region.
        intensity_image:
            Optional intensity image (unused but included for regionprops compatibility).

    Returns:
        Estimated volume in cubic pixels. Returns 0 if axis lengths cannot be computed.
    """
    # regionprops expects a labeled image, so convert the mask (0/1)
    labeled_mask = region_mask.astype(np.int64, copy=False)

    # Compute standard region properties on this mask
    props = ski.measure.regionprops(labeled_mask)[0]
    major_axis = float(props.axis_major_length)
    minor_axis = float(props.axis_minor_length)

    if major_axis == 0.0 or minor_axis == 0.0:
        return 0.0

    # Convert to semi-axes (regionprops returns full lengths)
    semi_major = major_axis / 2.0
    semi_minor = minor_axis / 2.0

    # Volume of prolate spheroid: (4/3) * π * a * b * b
    return (4.0 / 3.0) * np.pi * semi_major * semi_minor * semi_minor
