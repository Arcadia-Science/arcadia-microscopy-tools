import numpy as np
import pytest
import skimage as ski

from arcadia_microscopy_tools.channels import DAPI, FITC
from arcadia_microscopy_tools.masks import (
    DEFAULT_CELL_PROPERTY_NAMES,
    DEFAULT_INTENSITY_PROPERTY_NAMES,
    SegmentationMask,
    _extract_outlines_skimage,
)


def make_label_image(shape=(50, 50), cells=None):
    """Create a label image with disk-shaped cells.

    Args:
        shape: Image dimensions (rows, cols).
        cells: List of (cy, cx, radius) tuples. Defaults to a single cell at center.

    Returns:
        Int64 label image where each cell has a unique integer label (background=0).
    """
    label_image = np.zeros(shape, dtype=np.int64)
    if cells is None:
        cells = [(shape[0] // 2, shape[1] // 2, 8)]
    for label, (cy, cx, r) in enumerate(cells, start=1):
        rr, cc = ski.draw.disk((cy, cx), r, shape=shape)
        label_image[rr, cc] = label
    return label_image


def _make_mask(label_image):
    """Wrap a label image in a SegmentationMask with edge removal disabled."""
    return SegmentationMask(mask_image=label_image, remove_edge_cells=False)


def _make_mask_with_intensity(label_image):
    """SegmentationMask with two synthetic intensity channels (DAPI, FITC)."""
    rng = np.random.default_rng(42)
    dapi_img = rng.integers(100, 1000, size=label_image.shape).astype(np.uint16)
    fitc_img = rng.integers(0, 500, size=label_image.shape).astype(np.uint16)
    return SegmentationMask(
        mask_image=label_image,
        intensity_image_dict={DAPI: dapi_img, FITC: fitc_img},
        remove_edge_cells=False,
    )


@pytest.fixture
def interior_cell_image():
    """Single disk-shaped cell well inside a 50x50 image."""
    return make_label_image(shape=(50, 50), cells=[(25, 25, 8)])


@pytest.fixture
def multi_cell_image():
    """Two non-overlapping cells at opposite corners of a 60x60 image."""
    return make_label_image(shape=(60, 60), cells=[(15, 15, 6), (45, 45, 6)])


@pytest.fixture
def three_cell_image():
    """Three disk cells with distinct areas for filter testing.

    Radii 5, 8, 11 → approximate pixel areas ~78, ~201, ~380.
    All centers far enough from borders to survive edge-cell removal.
    """
    return make_label_image(
        shape=(80, 80),
        cells=[(20, 20, 5), (20, 60, 8), (60, 40, 11)],
    )


@pytest.fixture
def near_border_cell_image():
    """Cell whose bbox touches the image border (row 0) after tight crop.

    The cell center is at row 4 with radius 4, so its top edge sits at row 0.
    The bbox top is at row 0, meaning the old (unpadded) crop had no background
    above it — exactly the scenario the padding fix addresses.
    """
    return make_label_image(shape=(50, 50), cells=[(4, 25, 4)])


class TestExtractOutlinesSkimage:
    def test_returns_one_outline_per_cell(self, interior_cell_image, multi_cell_image):
        assert len(_extract_outlines_skimage(interior_cell_image)) == 1
        assert len(_extract_outlines_skimage(multi_cell_image)) == 2

    def test_outline_is_nonempty_2d_array(self, interior_cell_image):
        (outline,) = _extract_outlines_skimage(interior_cell_image)
        assert outline.ndim == 2
        assert outline.shape[1] == 2
        assert len(outline) > 0

    def test_outline_dtype_is_float(self, interior_cell_image):
        """find_contours returns sub-pixel float coordinates."""
        (outline,) = _extract_outlines_skimage(interior_cell_image)
        assert np.issubdtype(outline.dtype, np.floating)

    def test_coordinates_are_in_full_image_space(self, interior_cell_image):
        """Coordinates must be in (row, col) image space, not crop space."""
        h, w = interior_cell_image.shape
        (outline,) = _extract_outlines_skimage(interior_cell_image)
        assert outline[:, 0].min() >= 0
        assert outline[:, 0].max() < h
        assert outline[:, 1].min() >= 0
        assert outline[:, 1].max() < w

    def test_contour_surrounds_correct_cell(self, interior_cell_image):
        """Contour centroid should be near the cell center (25, 25)."""
        (outline,) = _extract_outlines_skimage(interior_cell_image)
        centroid = outline.mean(axis=0)
        assert np.linalg.norm(centroid - [25, 25]) < 2.0

    def test_contour_is_closed(self, interior_cell_image):
        """For interior cells, find_contours returns a closed loop (first == last)."""
        (outline,) = _extract_outlines_skimage(interior_cell_image)
        np.testing.assert_array_almost_equal(outline[0], outline[-1])

    def test_multiple_cells_ordered_by_label(self, multi_cell_image):
        """Outline at index i should correspond to label i+1."""
        outlines = _extract_outlines_skimage(multi_cell_image)
        centroid_0 = outlines[0].mean(axis=0)
        centroid_1 = outlines[1].mean(axis=0)
        assert np.linalg.norm(centroid_0 - [15, 15]) < 2.0
        assert np.linalg.norm(centroid_1 - [45, 45]) < 2.0

    def test_cell_near_border_produces_nonempty_contour(self, near_border_cell_image):
        """Cells whose tight bbox touches the image border should still get a contour.

        This was silently broken before the 1-pixel padding fix: the crop had no
        background above the cell, so find_contours left the contour open and the
        'main_contour' selection could yield a truncated or missing outline.
        """
        (outline,) = _extract_outlines_skimage(near_border_cell_image)
        assert len(outline) > 0

    def test_empty_outline_shape_contract(self):
        """When no contour is found, the placeholder must be shape (0, 2)."""
        # A single-pixel cell inside a tiny padded region: find_contours may or
        # may not detect a contour, but if it doesn't the shape must be (0, 2).
        label_image = np.zeros((5, 5), dtype=np.int64)
        label_image[2, 2] = 1
        outlines = _extract_outlines_skimage(label_image)
        assert len(outlines) == 1
        assert outlines[0].ndim == 2
        assert outlines[0].shape[1] == 2


class TestCellProperties:
    # --- output schema ---

    def test_default_keys_present(self, multi_cell_image):
        props = _make_mask(multi_cell_image).cell_properties
        expected = {
            p for p in DEFAULT_CELL_PROPERTY_NAMES if p not in ("centroid", "circularity", "volume")
        }
        expected |= {"centroid_y", "centroid_x", "circularity", "volume"}
        assert expected.issubset(props.keys())

    def test_centroid_renamed_from_skimage_format(self, multi_cell_image):
        """'centroid-0'/'centroid-1' must not appear; 'centroid_y'/'centroid_x' must."""
        props = _make_mask(multi_cell_image).cell_properties
        assert "centroid_y" in props
        assert "centroid_x" in props
        assert "centroid-0" not in props
        assert "centroid-1" not in props

    def test_one_value_per_cell(self, multi_cell_image):
        mask = _make_mask(multi_cell_image)
        props = mask.cell_properties
        for key, arr in props.items():
            assert len(arr) == mask.num_cells, f"Property '{key}' length mismatch"

    # --- value correctness ---

    def test_centroid_values_near_cell_centers(self, multi_cell_image):
        """Centroids should be close to the known disk centers."""
        props = _make_mask(multi_cell_image).cell_properties
        centers = np.array([[15.0, 15.0], [45.0, 45.0]])
        computed = np.column_stack([props["centroid_y"], props["centroid_x"]])
        for i, expected_center in enumerate(centers):
            assert np.linalg.norm(computed[i] - expected_center) < 2.0

    def test_circularity_in_valid_range(self, multi_cell_image):
        """Disk-shaped cells should have circularity close to 1.

        Keyword being should... perimeter estimate is bad, which causes circularity estimate
        to be off.
        """
        props = _make_mask(multi_cell_image).cell_properties
        circ = props["circularity"]
        assert np.all(circ >= 0.0)
        assert np.all(circ <= 1.0 + 0.1)  # add a fudge factor
        assert np.all(circ > 0.85), "Disk cells should have high circularity"

    def test_volume_positive(self, multi_cell_image):
        props = _make_mask(multi_cell_image).cell_properties
        assert np.all(props["volume"] > 0)

    # --- intensity properties ---

    def test_no_intensity_keys_without_intensity_images(self, multi_cell_image):
        props = _make_mask(multi_cell_image).cell_properties
        assert not any(k.startswith("intensity_") for k in props)

    def test_intensity_keys_with_channel_suffix_when_images_provided(self, multi_cell_image):
        """Intensity properties should be suffixed with the lowercased channel name."""
        props = _make_mask_with_intensity(multi_cell_image).cell_properties
        for intensity_prop in DEFAULT_INTENSITY_PROPERTY_NAMES:
            assert f"{intensity_prop}_dapi" in props
            assert f"{intensity_prop}_fitc" in props

    def test_intensity_values_have_correct_length(self, multi_cell_image):
        mask = _make_mask_with_intensity(multi_cell_image)
        props = mask.cell_properties
        for key in props:
            if key.endswith("_dapi") or key.endswith("_fitc"):
                assert len(props[key]) == mask.num_cells

    # --- custom property_names ---

    def test_custom_property_names_only_requested_keys_returned(self, multi_cell_image):
        """When property_names is customised, only those keys (plus centroid rename) appear."""
        mask = SegmentationMask(
            mask_image=multi_cell_image,
            remove_edge_cells=False,
            property_names=["label", "area"],
        )
        props = mask.cell_properties
        assert set(props.keys()) == {"label", "area"}

    def test_derived_properties_absent_when_not_requested(self, multi_cell_image):
        """circularity and volume must not appear when omitted from property_names."""
        mask = SegmentationMask(
            mask_image=multi_cell_image,
            remove_edge_cells=False,
            property_names=["label", "area"],
        )
        props = mask.cell_properties
        assert "circularity" not in props
        assert "volume" not in props

    def test_base_deps_not_leaked_when_only_circularity_requested(self, multi_cell_image):
        """area/perimeter added to compute circularity must be removed from output
        when the user did not explicitly request them."""
        mask = SegmentationMask(
            mask_image=multi_cell_image,
            remove_edge_cells=False,
            property_names=["circularity"],
        )
        props = mask.cell_properties
        assert "circularity" in props
        assert "area" not in props
        assert "perimeter" not in props


class TestFilter:
    # --- filtering behavior ---

    def test_min_value_removes_small_cells(self, three_cell_image):
        """min_value=150 should drop the r=5 cell (area≈78), keeping the two larger ones."""
        mask = _make_mask(three_cell_image)
        result = mask.filter("area", min_value=150)
        assert result.num_cells == 2

    def test_max_value_removes_large_cells(self, three_cell_image):
        """max_value=250 should drop the r=11 cell (area≈380), keeping the two smaller ones."""
        mask = _make_mask(three_cell_image)
        result = mask.filter("area", max_value=250)
        assert result.num_cells == 2

    def test_both_bounds_keeps_middle_cell(self, three_cell_image):
        """min=150, max=250 should keep only the r=8 cell (area≈201)."""
        mask = _make_mask(three_cell_image)
        result = mask.filter("area", min_value=150, max_value=250)
        assert result.num_cells == 1

    def test_correct_cells_retained_by_area(self, three_cell_image):
        """The retained cell should have an area value satisfying both bounds."""
        mask = _make_mask(three_cell_image)
        result = mask.filter("area", min_value=150, max_value=250)
        areas = result.cell_properties["area"]
        assert len(areas) == 1
        assert 150 <= areas[0] <= 250

    def test_filter_chaining(self, three_cell_image):
        """Applying two sequential filters should compose correctly."""
        mask = _make_mask(three_cell_image)
        step1 = mask.filter("area", min_value=100)  # drops r=5, keeps r=8 and r=11
        step2 = step1.filter("area", max_value=250)  # drops r=11, keeps r=8
        assert step2.num_cells == 1
        assert 150 <= step2.cell_properties["area"][0] <= 250

    # --- error handling ---

    def test_no_bounds_raises(self, multi_cell_image):
        with pytest.raises(ValueError, match="min_value or max_value"):
            _make_mask(multi_cell_image).filter("area")

    def test_unknown_property_raises(self, multi_cell_image):
        with pytest.raises(ValueError, match="not found"):
            _make_mask(multi_cell_image).filter("nonexistent_property", min_value=0)

    def test_no_cells_remaining_raises(self, multi_cell_image):
        """Filtering out all cells should raise ValueError."""
        with pytest.raises(ValueError, match="No cells remain"):
            _make_mask(multi_cell_image).filter("area", max_value=1)

    # --- config preservation ---

    def test_intensity_image_dict_preserved(self, multi_cell_image):
        mask = _make_mask_with_intensity(multi_cell_image)
        result = mask.filter("area", min_value=1)
        assert result.intensity_image_dict is not None
        assert mask.intensity_image_dict is not None
        assert set(result.intensity_image_dict.keys()) == set(mask.intensity_image_dict.keys())

    def test_outline_extractor_preserved(self, multi_cell_image):
        mask = SegmentationMask(
            mask_image=multi_cell_image,
            remove_edge_cells=False,
            outline_extractor="skimage",
        )
        result = mask.filter("area", min_value=1)
        assert result.outline_extractor == "skimage"

    def test_property_names_preserved(self, multi_cell_image):
        custom_props = ["label", "area", "perimeter"]
        mask = SegmentationMask(
            mask_image=multi_cell_image,
            remove_edge_cells=False,
            property_names=custom_props,
        )
        result = mask.filter("area", min_value=1)
        assert result.property_names == custom_props

    def test_intensity_property_names_preserved(self, multi_cell_image):
        mask = _make_mask_with_intensity(multi_cell_image)
        result = mask.filter("area", min_value=1)
        assert result.intensity_property_names == mask.intensity_property_names

    def test_remove_edge_cells_false_on_result(self, multi_cell_image):
        """Filtered mask must not re-run edge removal (labels are already clean)."""
        mask = _make_mask(multi_cell_image)
        result = mask.filter("area", min_value=1)
        assert result.remove_edge_cells is False
