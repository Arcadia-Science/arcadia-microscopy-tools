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

    Raises:
        ValueError: If no cells remain after processing (e.g. all cells were on the border).
    """
    label_image = mask_image
    if remove_edge_cells:
        label_image = ski.segmentation.clear_border(label_image)
        if label_image.max() == 0:
            raise ValueError(
                "No cells remain after removing edge cells. "
                "Try setting remove_edge_cells=False."
            )

    # Renumber existing labels to be consecutive starting from 1, preserving
    # cell identity. relabel_sequential is used rather than ski.measure.label
    # so that cells whose pixels happen to be non-contiguous are not silently
    # split into separate labels.
    label_image, _, _ = ski.segmentation.relabel_sequential(label_image)
    return label_image.astype(np.int64)


def _extract_outlines_cellpose(label_image: Int64Array) -> list[Float64Array]:
    """Extract cell outlines using Cellpose's outlines_list function.

    Args:
        label_image: 2D integer array where each cell has a unique label.

    Returns:
        List of arrays, one per cell, containing outline coordinates in (y, x) format.
    """
    outlines = outlines_list(label_image, multiprocessing=False)
    # Cellpose returns (x, y) coordinates, flip to (y, x) to match standard (row, col) format
    return [outline[:, [1, 0]] if len(outline) > 0 else outline for outline in outlines]


def _extract_outlines_skimage(label_image: Int64Array) -> list[Float64Array]:
    """Extract cell outlines using scikit-image's find_contours.

    Args:
        label_image: 2D integer array where each cell has a unique label.

    Returns:
        List of arrays, one per cell, containing outline coordinates in (y, x) format.
        Empty arrays are included for cells where no contours are found.
    """
    regions = ski.measure.regionprops(label_image)
    outlines = []
    for region in regions:
        # Crop to the cell's bounding box to avoid allocating a full-image mask per cell.
        minr, minc, maxr, maxc = region.bbox
        crop = (label_image[minr:maxr, minc:maxc] == region.label).astype(np.uint8)
        contours = ski.measure.find_contours(crop, level=0.5)
        if contours:
            main_contour = max(contours, key=len)
            # Shift contour coordinates back to full-image (row, col) space.
            main_contour += np.array([minr, minc])
            outlines.append(main_contour)
        else:
            # Include empty array to maintain alignment with cell labels.
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
            Defaults to "cellpose". In practice, cellpose is ~2x faster but skimage has been found
            to handle vertical contours better.
        property_names: List of property names to compute. If None, uses
            DEFAULT_CELL_PROPERTY_NAMES.
        intensity_property_names: List of intensity property names to compute.
            If None, uses DEFAULT_INTENSITY_PROPERTY_NAMES when intensity_image_dict is provided.
    """

    mask_image: BoolArray | Int64Array
    intensity_image_dict: Mapping[Channel, UInt16Array] | None = None
    remove_edge_cells: bool = True
    outline_extractor: Literal["cellpose", "skimage"] = "cellpose"
    # Accepts None at construction time; always a list[str] after __post_init__.
    property_names: list[str] | None = field(default=None)
    # Accepts None at construction time; always a list[str] after __post_init__.
    intensity_property_names: list[str] | None = field(default=None)

    # Core fields that must not be mutated after initialisation. cached_property
    # writes directly to instance.__dict__, bypassing __setattr__, so it is unaffected.
    _IMMUTABLE_FIELDS: frozenset[str] = field(
        default=frozenset({
            "mask_image",
            "intensity_image_dict",
            "remove_edge_cells",
            "outline_extractor",
            "property_names",
            "intensity_property_names",
        }),
        init=False,
        repr=False,
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialized", False) and name in self._IMMUTABLE_FIELDS:
            raise AttributeError(
                f"Cannot modify '{name}' after SegmentationMask is initialized. "
                "Create a new instance instead."
            )
        super().__setattr__(name, value)

    def __post_init__(self):
        """Validate inputs and set defaults."""
        # Validate mask_image
        if not isinstance(self.mask_image, np.ndarray):
            raise TypeError("mask_image must be a numpy array")
        if self.mask_image.ndim != 2:
            raise ValueError("mask_image must be a 2D array")
        if np.any(self.mask_image < 0):
            raise ValueError("mask_image must have non-negative values")
        if self.mask_image.max() == 0:
            raise ValueError("mask_image contains no cells (all values are 0)")

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
            # Shallow-copy the dict so that channel key changes in one instance
            # (e.g. after filter()) do not affect another. The underlying numpy
            # arrays are shared by reference; they are not copied.
            self.intensity_image_dict = dict(self.intensity_image_dict)

        # Set default property names if none provided
        if self.property_names is None:
            self.property_names = DEFAULT_CELL_PROPERTY_NAMES.copy()

        # Set default intensity property names if intensity images provided
        if self.intensity_property_names is None:
            if self.intensity_image_dict:
                self.intensity_property_names = DEFAULT_INTENSITY_PROPERTY_NAMES.copy()
            else:
                self.intensity_property_names = []

        # Lock core fields against post-init mutation.
        object.__setattr__(self, "_initialized", True)

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

        Note:
            The cellpose method is ~2x faster in general but skimage handles
            vertically oriented cells/outlines better.
        """
        if self.outline_extractor == "cellpose":
            return _extract_outlines_cellpose(self.label_image)
        else:  # must be "skimage" due to Literal type
            return _extract_outlines_skimage(self.label_image)

    @cached_property
    def cell_properties(self) -> dict[str, ScalarArray]:
        """Extract cell property values using regionprops.

        Extracts both morphological properties (area, perimeter, etc.) and intensity-based
        properties (mean, max, min intensity) for each channel if intensity images are provided.

        For multichannel intensity images, property names are suffixed with the channel name:
        - DAPI: "intensity_mean_DAPI"
        - FITC: "intensity_max_FITC"

        Returns:
            Dictionary mapping property names to arrays of values (one per cell).
        """
        assert self.property_names is not None  # guaranteed by __post_init__

        # circularity and volume are derived quantities not known to skimage.
        # Compute them post-hoc from already-fetched scalar arrays to avoid
        # calling regionprops a second time per cell inside extra_properties callbacks.
        needs_circularity = "circularity" in self.property_names
        needs_volume = "volume" in self.property_names

        # Build the skimage-compatible property list (strip derived names).
        skimage_props = [p for p in self.property_names if p not in ("circularity", "volume")]

        # Temporarily add any base properties needed for the derived computations
        # if the user did not explicitly request them.
        added_props: set[str] = set()
        for dep in (["area", "perimeter"] if needs_circularity else []):
            if dep not in skimage_props:
                skimage_props.append(dep)
                added_props.add(dep)
        for dep in (["axis_major_length", "axis_minor_length"] if needs_volume else []):
            if dep not in skimage_props:
                skimage_props.append(dep)
                added_props.add(dep)

        # Compute cell properties
        properties = ski.measure.regionprops_table(
            self.label_image,
            properties=skimage_props,
        )

        # Derive circularity: (4π·area) / perimeter², clamped to 0 when perimeter == 0.
        if needs_circularity:
            area = properties["area"]
            perimeter = properties["perimeter"]
            properties["circularity"] = np.where(
                perimeter > 0, (4.0 * np.pi * area) / (perimeter**2), 0.0
            )

        # Derive volume: prolate spheroid (4/3)π·a·b² from semi-axes.
        if needs_volume:
            a = properties["axis_major_length"] / 2.0
            b = properties["axis_minor_length"] / 2.0
            properties["volume"] = np.where(
                (a > 0) & (b > 0), (4.0 / 3.0) * np.pi * a * b * b, 0.0
            )

        # Remove base properties that were added solely to support derived computations.
        for prop in added_props:
            properties.pop(prop, None)

        if "centroid-0" in properties:
            properties["centroid_y"] = properties.pop("centroid-0")
        if "centroid-1" in properties:
            properties["centroid_x"] = properties.pop("centroid-1")

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
                    properties[f"{prop_name}_{channel.name.lower()}"] = prop_values

        return properties

    @cached_property
    def centroids_yx(self) -> Float64Array:
        """Get cell centroids as (y, x) coordinates.

        Returns:
            Array of shape (num_cells, 2) with centroid coordinates.
            Each row is [y_coordinate, x_coordinate] for one cell.
            Returns empty (0, 2) array with warning if "centroid" not in property_names.

        Raises:
            ValueError: If no cells are found in the mask.
        """
        assert self.property_names is not None  # guaranteed by __post_init__
        if "centroid" not in self.property_names:
            warnings.warn(
                "Centroid property not available. Include 'centroid' in property_names "
                "to get centroid coordinates. Returning empty array.",
                UserWarning,
                stacklevel=2,
            )
            return np.array([]).reshape(0, 2)

        yc = self.cell_properties["centroid_y"]
        xc = self.cell_properties["centroid_x"]
        return np.array([yc, xc], dtype=float).T

    def filter(
        self,
        property_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> SegmentationMask:
        """Return a new SegmentationMask with cells removed based on a property threshold.

        Args:
            property_name: Name of the property to filter on. Must be a key in cell_properties.
            min_value: Minimum value (inclusive). Cells with values below this are removed.
            max_value: Maximum value (inclusive). Cells with values above this are removed.

        Returns:
            A new SegmentationMask containing only cells that pass the filter.

        Raises:
            ValueError: If neither min_value nor max_value is provided.
            ValueError: If property_name is not found in cell_properties.
            ValueError: If no cells remain after filtering.
        """
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be provided.")

        if property_name not in self.cell_properties:
            raise ValueError(
                f"Property '{property_name}' not found. "
                f"Available properties: {list(self.cell_properties.keys())}"
            )

        values = self.cell_properties[property_name]
        # Labels are consecutive 1..num_cells after _process_mask
        labels = np.arange(1, self.num_cells + 1)

        keep = np.ones(len(labels), dtype=bool)
        if min_value is not None:
            keep &= values >= min_value
        if max_value is not None:
            keep &= values <= max_value

        labels_to_keep = labels[keep]
        new_label_image = np.where(
            np.isin(self.label_image, labels_to_keep),
            self.label_image,
            0,
        ).astype(np.int64)

        if new_label_image.max() == 0:
            raise ValueError(
                f"No cells remain after filtering '{property_name}' "
                f"with min={min_value}, max={max_value}."
            )

        return SegmentationMask(
            mask_image=new_label_image,
            intensity_image_dict=self.intensity_image_dict,
            remove_edge_cells=False,
            outline_extractor=self.outline_extractor,
            property_names=self.property_names,
            intensity_property_names=self.intensity_property_names,
        )

    def convert_properties_to_microns(
        self,
        pixel_size_um: float,
    ) -> dict[str, ScalarArray]:
        """Convert cell properties from pixels to microns.

        Applies appropriate scaling factors based on the dimensionality of each property:
            - Linear measurements (1D): multiplied by pixel_size_um, keys suffixed with "_um"
            - Area measurements (2D): multiplied by pixel_size_um², keys suffixed with "_um2"
            - Volume measurements (3D): multiplied by pixel_size_um³, keys suffixed with "_um3"
            - Dimensionless properties: unchanged, keys unchanged

        Args:
            pixel_size_um: Pixel size in microns.

        Returns:
            Dictionary with keys renamed to include units and values
            converted to micron units where applicable.

        Note:
            Properties like 'label', 'circularity', 'eccentricity', 'solidity', and 'orientation'
            are dimensionless and remain unchanged. Intensity properties (intensity_mean,
            intensity_max, etc.) are also dimensionless and remain unchanged. Centroid coordinates
            (centroid_y, centroid_x) remain in pixel coordinates as they represent image positions.
            Tensor properties (inertia_tensor, inertia_tensor_eigvals) are scaled as 2D quantities
            (pixel_size_um²) and suffixed with "_um2".
        """
        # Define which properties need which scaling
        linear_properties = {"perimeter", "axis_major_length", "axis_minor_length"}
        area_properties = {"area", "area_convex"}
        volume_properties = {"volume"}
        tensor_properties = {"inertia_tensor", "inertia_tensor_eigvals"}

        converted = {}
        for prop_name, prop_values in self.cell_properties.items():
            if prop_name in linear_properties:
                converted[f"{prop_name}_um"] = prop_values * pixel_size_um
            elif prop_name in area_properties:
                converted[f"{prop_name}_um2"] = prop_values * (pixel_size_um**2)
            elif prop_name in volume_properties:
                converted[f"{prop_name}_um3"] = prop_values * (pixel_size_um**3)
            elif prop_name in tensor_properties:
                converted[f"{prop_name}_um2"] = prop_values * (pixel_size_um**2)
            else:
                # Intensity-related, dimensionless, or label - no conversion
                converted[prop_name] = prop_values

        return converted


