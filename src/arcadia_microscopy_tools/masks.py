from __future__ import annotations
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import skimage as ski
from cellpose.utils import outlines_list

from .channels import Channel
from .typing import BoolArray, Float64Array, Int64Array, ScalarArray, UInt16Array

DEFAULT_CELL_PROPERTY_NAMES = [
    "label",
    "centroid",
    "volume",
    "area",
    "area_convex",
    "perimeter",
    "eccentricity",
    "circularity",
    "solidity",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
]

DEFAULT_INTENSITY_PROPERTY_NAMES = [
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "intensity_std",
]

OutlineExtractorMethod = Literal["cellpose", "skimage"]


def _process_mask(
    mask_image: BoolArray | Int64Array,
    remove_edge_cells: bool,
) -> Int64Array:
    """Process a mask image by optionally removing edge cells and ensuring consecutive labels.

    Args:
        mask_image: Input mask array where each cell has a unique label.
        remove_edge_cells: Whether to remove cells touching image borders.

    Returns:
        Processed label image with consecutive labels starting from 1.
    """
    label_image = mask_image
    if remove_edge_cells:
        label_image = ski.segmentation.clear_border(label_image)

    # Ensure consecutive labels
    label_image = ski.measure.label(label_image).astype(np.int64)  # type: ignore
    return label_image


def _extract_outlines_cellpose(label_image: Int64Array) -> list[Float64Array]:
    """Extract cell outlines using Cellpose's outlines_list function.

    Args:
        label_image: 2D integer array where each cell has a unique label.

    Returns:
        List of arrays, one per cell, containing outline coordinates in (y, x) format.
    """
    return outlines_list(label_image, multiprocessing=False)


def _extract_outlines_skimage(label_image: Int64Array) -> list[Float64Array]:
    """Extract cell outlines using scikit-image's find_contours.

    Args:
        label_image: 2D integer array where each cell has a unique label.

    Returns:
        List of arrays, one per cell, containing outline coordinates in (y, x) format.
        Empty arrays are included for cells where no contours are found.
    """
    # Get unique cell IDs (excluding background)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]

    outlines = []
    for cell_id in unique_labels:
        cell_mask = (label_image == cell_id).astype(np.uint8)
        contours = ski.measure.find_contours(cell_mask, level=0.5)
        if contours:
            main_contour = max(contours, key=len)
            # Flip from (x, y) to (y, x) to match cellpose format
            main_contour = main_contour[:, [1, 0]]
            outlines.append(main_contour)
        else:
            # Include empty array to maintain alignment with cell labels
            outlines.append(np.array([]).reshape(0, 2))
    return outlines


@dataclass
class SegmentationMask:
    """Container for segmentation mask data and feature extraction.

    Args:
        mask_image: 2D integer or boolean array where each cell has a unique label (background=0).
        intensity_image_dict: Optional dict mapping Channel instances to 2D intensity arrays.
            Each intensity array must have the same shape as mask_image. Channel names will be used
            as suffixes for intensity properties. Example:
                {DAPI: array, FITC: array}
        remove_edge_cells: Whether to remove cells touching image borders. Defaults to True.
        outline_extractor: Outline extraction method ("cellpose" or "skimage").
            Defaults to "cellpose".
        property_names: List of property names to compute. If None, uses
            DEFAULT_CELL_PROPERTY_NAMES.
        intensity_property_names: List of intensity property names to compute.
            If None, uses DEFAULT_INTENSITY_PROPERTY_NAMES when intensity_image_dict is provided.
    """

    mask_image: BoolArray | Int64Array
    intensity_image_dict: Mapping[Channel, UInt16Array] | None = None
    remove_edge_cells: bool = True
    outline_extractor: OutlineExtractorMethod = "cellpose"
    property_names: list[str] | None = field(default=None)
    intensity_property_names: list[str] | None = field(default=None)

    def __post_init__(self):
        """Validate inputs and set defaults."""
        # Validate mask_image
        if not isinstance(self.mask_image, np.ndarray):
            raise TypeError("mask_image must be a numpy array")
        if self.mask_image.ndim != 2:
            raise ValueError("mask_image must be a 2D array")
        if self.mask_image.min() < 0:
            raise ValueError("mask_image must have non-negative values")

        # Validate intensity_image dict if provided
        if self.intensity_image_dict is not None:
            if not isinstance(self.intensity_image_dict, Mapping):
                raise TypeError("intensity_image_dict must be a Mapping of channels to 2D arrays")
            for channel, intensities in self.intensity_image_dict.items():
                if not isinstance(intensities, np.ndarray):
                    raise TypeError(f"Intensity image for '{channel.name}' must be a numpy array")
                if intensities.ndim != 2:
                    raise ValueError(f"Intensity image for '{channel.name}' must be 2D")
                if intensities.shape != self.mask_image.shape:
                    raise ValueError(
                        f"Intensity image for '{channel.name}' must have same shape as mask_image"
                    )

        # Set default property names if none provided
        if self.property_names is None:
            self.property_names = DEFAULT_CELL_PROPERTY_NAMES.copy()

        # Set default intensity property names if intensity images provided
        if self.intensity_property_names is None:
            if self.intensity_image_dict:
                self.intensity_property_names = DEFAULT_INTENSITY_PROPERTY_NAMES.copy()
            else:
                self.intensity_property_names = []

    @cached_property
    def label_image(self) -> Int64Array:
        """Get processed label image with consecutive labels.

        Returns:
            2D integer array with consecutive cell labels starting from 1 (background=0).
            Edge cells removed if remove_edge_cells=True.
        """
        return _process_mask(self.mask_image, self.remove_edge_cells)

    @cached_property
    def num_cells(self) -> int:
        """Get the number of cells in the mask.

        Returns:
            Integer count of cells (maximum label value in label_image).
        """
        return int(self.label_image.max())

    @cached_property
    def cell_outlines(self) -> list[Float64Array]:
        """Extract cell outlines using the configured outline extractor.

        Returns:
            List of arrays, one per cell, containing outline coordinates in (y, x) format.
            Returns empty list if no cells found.
        """
        if self.num_cells == 0:
            return []

        if self.outline_extractor == "cellpose":
            return _extract_outlines_cellpose(self.label_image)
        else:  # Must be "skimage" due to Literal type
            return _extract_outlines_skimage(self.label_image)

    @cached_property
    def cell_properties(self) -> dict[str, ScalarArray]:
        """Extract cell property values using regionprops.

        Extracts both morphological properties (area, perimeter, etc.) and intensity-based
        properties (mean, max, min intensity) for each channel if intensity images are provided.

        For multichannel intensity images, property names are suffixed with the channel name:
        - DAPI: "intensity_mean_DAPI"
        - FITC: "intensity_mean_FITC"

        Returns:
            Dictionary mapping property names to arrays of values (one per cell).
            Returns empty dict if no cells found.
        """
        if self.num_cells == 0:
            empty_props = (
                {property_name: np.array([]) for property_name in self.property_names}
                if self.property_names
                else {}
            )
            # Add empty intensity properties if requested
            if self.intensity_image_dict and self.intensity_property_names:
                for channel in self.intensity_image_dict:
                    for prop_name in self.intensity_property_names:
                        empty_props[f"{prop_name}_{channel.name}"] = np.array([])
            return empty_props

        # Extract morphological properties (no intensity image needed)
        # Only compute extra properties if explicitly requested
        extra_props = []
        if self.property_names and "circularity" in self.property_names:
            extra_props.append(circularity)
        if self.property_names and "volume" in self.property_names:
            extra_props.append(volume)

        properties = ski.measure.regionprops_table(
            self.label_image,
            properties=self.property_names,
            extra_properties=extra_props,
        )

        # Extract intensity properties for each channel
        if self.intensity_image_dict and self.intensity_property_names:
            for channel, intensities in self.intensity_image_dict.items():
                channel_props = ski.measure.regionprops_table(
                    self.label_image,
                    intensity_image=intensities,
                    properties=self.intensity_property_names,
                )
                # Add channel suffix to property names
                for prop_name, prop_values in channel_props.items():
                    properties[f"{prop_name}_{channel.name}"] = prop_values

        return properties

    @cached_property
    def centroids_yx(self) -> Float64Array:
        """Get cell centroids as (y, x) coordinates.

        Returns:
            Array of shape (num_cells, 2) with centroid coordinates.
            Each row is [y_coordinate, x_coordinate] for one cell.
            Returns empty (0, 2) array with warning if "centroid" not in property_names.
        """
        if self.property_names and "centroid" not in self.property_names:
            warnings.warn(
                "Centroid property not available. Include 'centroid' in property_names "
                "to get centroid coordinates. Returning empty array.",
                UserWarning,
                stacklevel=2,
            )
            return np.array([]).reshape(0, 2)

        yc = self.cell_properties["centroid-0"]
        xc = self.cell_properties["centroid-1"]
        return np.array([yc, xc], dtype=float).T

    def convert_properties_to_microns(
        self,
        pixel_size_um: float,
    ) -> dict[str, ScalarArray]:
        """Convert cell properties from pixels to microns.

        Applies appropriate scaling factors based on the dimensionality of each property:
            - Linear measurements (1D): multiplied by pixel_size_um
            - Area measurements (2D): multiplied by pixel_size_um²
            - Volume measurements (3D): multiplied by pixel_size_um³
            - Dimensionless properties: unchanged

        Args:
            pixel_size_um: Pixel size in microns.

        Returns:
            Dictionary with the same keys as cell_properties but with values
            converted to micron units where applicable.

        Note:
            Properties like 'label', 'circularity', 'eccentricity', 'solidity',
            and 'orientation' are dimensionless and remain unchanged.
            Intensity properties (intensity_mean, intensity_max, etc.) are also
            dimensionless and remain unchanged.
            Tensor properties (inertia_tensor, inertia_tensor_eigvals) are scaled
            as 2D quantities (pixel_size_um²).
        """
        # Define which properties need which scaling
        linear_properties = {
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "centroid-0",
            "centroid-1",
        }
        area_properties = {"area", "area_convex"}
        volume_properties = {"volume"}
        tensor_properties = {"inertia_tensor", "inertia_tensor_eigvals"}

        converted = {}
        for prop_name, prop_values in self.cell_properties.items():
            if prop_name in linear_properties:
                converted[prop_name] = prop_values * pixel_size_um
            elif prop_name in area_properties:
                converted[prop_name] = prop_values * (pixel_size_um**2)
            elif prop_name in volume_properties:
                converted[prop_name] = prop_values * (pixel_size_um**3)
            elif prop_name in tensor_properties:
                converted[prop_name] = prop_values * (pixel_size_um**2)
            else:
                # Intensity-related, dimensionless, or label - no conversion
                converted[prop_name] = prop_values

        return converted


def circularity(region_mask: BoolArray) -> float:
    """Calculate the circularity of a cell region.

    Circularity is a shape metric that quantifies how close a shape is to a perfect circle.
    It is computed as (4π * area) / perimeter², ranging from 0 to 1, where 1 represents
    a perfect circle and lower values indicate more elongated or irregular shapes.

    Args:
        region_mask: Boolean mask of the cell region.

    Returns:
        Circularity value between 0 and 1. Returns 0 if perimeter is zero.
    """
    # regionprops expects a labeled image, so convert the mask (0/1)
    labeled_mask = region_mask.astype(np.int64)

    # Compute standard region properties on this mask
    props = ski.measure.regionprops(labeled_mask)[0]
    area = float(props.area)
    perimeter = float(props.perimeter)

    if perimeter == 0.0:
        return 0.0

    return (4.0 * np.pi * area) / (perimeter**2)


def volume(region_mask: BoolArray) -> float:
    """Estimate the volume of a cell region.

    Volume is estimated by fitting an ellipse to the cell region and treating it as
    a prolate spheroid (ellipsoid of revolution). The ellipsoid is formed by rotating
    the fitted ellipse around its major axis, with volume = (4/3)π * a * b^2, where
    a is the semi-major axis and b is the semi-minor axis.

    Args:
        region_mask: Boolean mask of the cell region.

    Returns:
        Estimated volume in cubic pixels. Returns 0 if axis lengths cannot be computed.
    """
    # regionprops expects a labeled image, so convert the mask (0/1)
    labeled_mask = region_mask.astype(np.int64)

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
