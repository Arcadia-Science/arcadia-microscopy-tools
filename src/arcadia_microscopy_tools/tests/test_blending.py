import numpy as np
import pytest
from arcadia_pycolor import HexCode
from matplotlib.colors import LinearSegmentedColormap

from arcadia_microscopy_tools.blending import (
    Layer,
    alpha_blend,
    colorize,
    create_sequential_overlay,
    overlay_channels,
)
from arcadia_microscopy_tools.channels import Channel

BLUE = HexCode("blue", "#0000FF")
GREEN = HexCode("green", "#00FF00")

CHAN_BLUE = Channel(name="Blue", color=BLUE)
CHAN_GREEN = Channel(name="Green", color=GREEN)


@pytest.fixture
def background():
    return np.full((4, 4), 0.5, dtype=np.float64)


@pytest.fixture
def ones_layer():
    return np.ones((4, 4), dtype=np.float64)


@pytest.fixture
def zeros_layer():
    return np.zeros((4, 4), dtype=np.float64)


# ---------- Layer validation ----------


class TestLayer:
    def test_valid_layer(self, ones_layer):
        layer = Layer(CHAN_BLUE, ones_layer)
        assert layer.opacity == 1.0
        assert layer.transparent is True

    def test_non_2d_intensities_raises(self):
        with pytest.raises(ValueError, match="Expected 2D"):
            Layer(CHAN_BLUE, np.ones((4, 4, 3), dtype=np.float64))

    def test_opacity_below_zero_raises(self, ones_layer):
        with pytest.raises(ValueError, match="Opacity must be in"):
            Layer(CHAN_BLUE, ones_layer, opacity=-0.1)

    def test_opacity_above_one_raises(self, ones_layer):
        with pytest.raises(ValueError, match="Opacity must be in"):
            Layer(CHAN_BLUE, ones_layer, opacity=1.5)

    def test_opacity_boundary_values(self, ones_layer):
        Layer(CHAN_BLUE, ones_layer, opacity=0.0)
        Layer(CHAN_BLUE, ones_layer, opacity=1.0)


# ---------- alpha_blend ----------


class TestAlphaBlend:
    def test_alpha_zero_returns_background(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.9, dtype=np.float64)
        alpha = np.zeros((2, 2, 1), dtype=np.float64)
        result = alpha_blend(bg, fg, alpha)
        np.testing.assert_allclose(result, bg)

    def test_alpha_one_returns_foreground(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.9, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = alpha_blend(bg, fg, alpha)
        np.testing.assert_allclose(result, fg)

    def test_alpha_half_is_midpoint(self):
        bg = np.zeros((2, 2, 3), dtype=np.float64)
        fg = np.ones((2, 2, 3), dtype=np.float64)
        alpha = np.full((2, 2, 1), 0.5, dtype=np.float64)
        result = alpha_blend(bg, fg, alpha)
        np.testing.assert_allclose(result, 0.5)

    def test_output_clipped_to_unit_range(self):
        bg = np.full((2, 2, 3), 1.1, dtype=np.float64)
        fg = np.full((2, 2, 3), 1.2, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = alpha_blend(bg, fg, alpha)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


# ---------- colorize ----------


class TestColorize:
    def test_output_shape(self):
        intensities = np.random.rand(8, 8).astype(np.float64)
        cmap = LinearSegmentedColormap.from_list("test", ["black", "white"])
        result = colorize(intensities, cmap)
        assert result.shape == (8, 8, 4)

    def test_output_dtype(self):
        intensities = np.zeros((4, 4), dtype=np.float64)
        cmap = LinearSegmentedColormap.from_list("test", ["black", "white"])
        result = colorize(intensities, cmap)
        assert result.dtype == np.float64

    def test_non_2d_raises(self):
        cmap = LinearSegmentedColormap.from_list("test", ["black", "white"])
        with pytest.raises(ValueError, match="Expected 2D"):
            colorize(np.ones((4,), dtype=np.float64), cmap)
        with pytest.raises(ValueError, match="Expected 2D"):
            colorize(np.ones((4, 4, 3), dtype=np.float64), cmap)


# ---------- create_sequential_overlay ----------


class TestCreateSequentialOverlay:
    def test_no_layers_returns_gray_rgb(self, background):
        result = create_sequential_overlay(background, [])
        assert result.shape == (4, 4, 3)
        np.testing.assert_allclose(result, 0.5)

    def test_non_2d_background_raises(self):
        bg_3d = np.ones((4, 4, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="Expected 2D background"):
            create_sequential_overlay(bg_3d, [])

    def test_output_shape(self, background, ones_layer):
        layers = [Layer(CHAN_BLUE, ones_layer)]
        result = create_sequential_overlay(background, layers)
        assert result.shape == (4, 4, 3)

    def test_output_in_unit_range(self, background, ones_layer):
        layers = [
            Layer(CHAN_BLUE, ones_layer),
            Layer(CHAN_GREEN, ones_layer, opacity=0.5),
        ]
        result = create_sequential_overlay(background, layers)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_opacity_preserves_background(self, background, ones_layer):
        result = create_sequential_overlay(background, [Layer(CHAN_BLUE, ones_layer, opacity=0.0)])
        expected = np.full((4, 4, 3), 0.5, dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------- overlay_channels ----------


class TestOverlayChannels:
    def test_basic_overlay(self, background, ones_layer):
        result = overlay_channels(background, {CHAN_BLUE: ones_layer})
        assert result.shape == (4, 4, 3)

    def test_empty_channels_returns_gray_rgb(self, background):
        result = overlay_channels(background, {})
        assert result.shape == (4, 4, 3)
        np.testing.assert_allclose(result, 0.5)

    def test_multiple_channels(self, background, ones_layer):
        result = overlay_channels(
            background,
            {CHAN_BLUE: ones_layer, CHAN_GREEN: ones_layer},
        )
        assert result.shape == (4, 4, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_opaque_mode(self, background, ones_layer):
        result = overlay_channels(background, {CHAN_BLUE: ones_layer}, transparent=False)
        assert result.shape == (4, 4, 3)
