import numpy as np
import pytest
import skimage as ski

from arcadia_microscopy_tools.masks import (
    _extract_outlines_cellpose,
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


@pytest.fixture
def interior_cell_image():
    """Single disk-shaped cell well inside a 50x50 image."""
    return make_label_image(shape=(50, 50), cells=[(25, 25, 8)])


@pytest.fixture
def multi_cell_image():
    """Two non-overlapping cells at opposite corners of a 60x60 image."""
    return make_label_image(shape=(60, 60), cells=[(15, 15, 6), (45, 45, 6)])


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


class TestSkimageVsCellpose:
    def test_same_outline_count(self, interior_cell_image, multi_cell_image):
        for img in (interior_cell_image, multi_cell_image):
            ski_out = _extract_outlines_skimage(img)
            cp_out = _extract_outlines_cellpose(img)
            assert len(ski_out) == len(cp_out)

    def test_nonempty_agreement(self, multi_cell_image):
        """Both methods should agree on which cells have detectable outlines."""
        ski_out = _extract_outlines_skimage(multi_cell_image)
        cp_out = _extract_outlines_cellpose(multi_cell_image)
        for s, c in zip(ski_out, cp_out, strict=True):
            assert (len(s) > 0) == (len(c) > 0)

    def test_outlines_trace_same_region(self, interior_cell_image):
        """Centroids from both methods should be close to the same cell center."""
        (ski_outline,) = _extract_outlines_skimage(interior_cell_image)
        (cp_outline,) = _extract_outlines_cellpose(interior_cell_image)
        ski_centroid = ski_outline.mean(axis=0)
        cp_centroid = cp_outline.mean(axis=0)
        assert np.linalg.norm(ski_centroid - cp_centroid) < 3.0

    def test_empty_outlines_have_shape_0_2(self, multi_cell_image):
        """Empty placeholder outlines must be (0, 2) for both methods."""
        for outlines in (
            _extract_outlines_skimage(multi_cell_image),
            _extract_outlines_cellpose(multi_cell_image),
        ):
            for outline in outlines:
                if len(outline) == 0:
                    assert outline.shape == (0, 2)
