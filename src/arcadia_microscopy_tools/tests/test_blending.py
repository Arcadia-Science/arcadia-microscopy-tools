import numpy as np
import pytest

from arcadia_microscopy_tools.blending import (
    BlendMode,
    Layer,
    _blend_additive,
    _blend_alpha,
    _build_colormap,
    _gray_to_rgb,
    create_overlay,
    overlay_channels,
)
from arcadia_microscopy_tools.channels import Channel

CHAN_BLUE = Channel("Blue", "#0000FF")
CHAN_GREEN = Channel("Green", "#00FF00")


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
        assert layer.zero_transparent is True
        assert layer.blend_mode is BlendMode.ADDITIVE

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

    def test_out_of_range_intensities_warns_and_clips(self):
        raw = np.array([[0.0, 2.0], [-0.5, 0.5]], dtype=np.float64)
        with pytest.warns(match="outside \\[0, 1\\]"):
            layer = Layer(CHAN_BLUE, raw)
        assert float(layer.intensities.min()) >= 0.0
        assert float(layer.intensities.max()) <= 1.0

    def test_in_range_intensities_no_warning(self, ones_layer):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Layer(CHAN_BLUE, ones_layer)


# ---------- Blend functions ----------


class TestBlendAlpha:
    def test_alpha_zero_returns_background(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.9, dtype=np.float64)
        alpha = np.zeros((2, 2, 1), dtype=np.float64)
        result = _blend_alpha(bg, fg, alpha)
        np.testing.assert_allclose(result, bg)

    def test_alpha_one_returns_foreground(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.9, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = _blend_alpha(bg, fg, alpha)
        np.testing.assert_allclose(result, fg)

    def test_alpha_half_is_midpoint(self):
        bg = np.zeros((2, 2, 3), dtype=np.float64)
        fg = np.ones((2, 2, 3), dtype=np.float64)
        alpha = np.full((2, 2, 1), 0.5, dtype=np.float64)
        result = _blend_alpha(bg, fg, alpha)
        np.testing.assert_allclose(result, 0.5)

    def test_output_clipped_to_unit_range(self):
        bg = np.full((2, 2, 3), 1.1, dtype=np.float64)
        fg = np.full((2, 2, 3), 1.2, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = _blend_alpha(bg, fg, alpha)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


class TestBlendAdditive:
    def test_alpha_zero_returns_background(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.9, dtype=np.float64)
        alpha = np.zeros((2, 2, 1), dtype=np.float64)
        result = _blend_additive(bg, fg, alpha)
        np.testing.assert_allclose(result, bg)

    def test_adds_foreground_contribution(self):
        bg = np.full((2, 2, 3), 0.3, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.2, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = _blend_additive(bg, fg, alpha)
        np.testing.assert_allclose(result, 0.5)

    def test_output_clipped_to_unit_range(self):
        bg = np.full((2, 2, 3), 0.8, dtype=np.float64)
        fg = np.full((2, 2, 3), 0.8, dtype=np.float64)
        alpha = np.ones((2, 2, 1), dtype=np.float64)
        result = _blend_additive(bg, fg, alpha)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_is_commutative(self, background, ones_layer):
        """Additive blending should not depend on layer order."""
        layers_ab = [
            Layer(CHAN_BLUE, ones_layer * 0.3),
            Layer(CHAN_GREEN, ones_layer * 0.5),
        ]
        layers_ba = [
            Layer(CHAN_GREEN, ones_layer * 0.5),
            Layer(CHAN_BLUE, ones_layer * 0.3),
        ]
        result_ab = create_overlay(background, layers_ab)
        result_ba = create_overlay(background, layers_ba)
        np.testing.assert_allclose(result_ab, result_ba, atol=1e-10)


# ---------- Colormap ----------


class TestBuildColormap:
    def test_transparent_and_opaque_differ(self):
        cmap_t = _build_colormap("#FF0000", True)
        cmap_o = _build_colormap("#FF0000", False)
        vals = np.linspace(0, 1, 5)
        rgba_t = cmap_t(vals)
        rgba_o = cmap_o(vals)
        assert not np.allclose(rgba_t, rgba_o)

    def test_transparent_zero_has_zero_alpha(self):
        cmap = _build_colormap("#FF0000", True)
        rgba = cmap(np.array([0.0]))
        assert float(rgba[0, 3]) == pytest.approx(0.0)

    def test_opaque_zero_has_full_alpha(self):
        cmap = _build_colormap("#FF0000", False)
        rgba = cmap(np.array([0.0]))
        assert float(rgba[0, 3]) == pytest.approx(1.0)

    def test_caching(self):
        a = _build_colormap("#0000FF", True)
        b = _build_colormap("#0000FF", True)
        assert a is b


# ---------- Utilities ----------


class TestGrayToRgb:
    def test_shape(self):
        gray = np.zeros((3, 5), dtype=np.float64)
        result = _gray_to_rgb(gray)
        assert result.shape == (3, 5, 3)

    def test_values_broadcast(self):
        gray = np.full((2, 2), 0.7, dtype=np.float64)
        result = _gray_to_rgb(gray)
        np.testing.assert_allclose(result, 0.7)


# ---------- create_overlay ----------


class TestCreateOverlay:
    def test_no_layers_returns_gray_rgb(self, background):
        result = create_overlay(background, [])
        assert result.shape == (4, 4, 3)
        np.testing.assert_allclose(result, 0.5)

    def test_non_2d_background_raises(self):
        bg_3d = np.ones((4, 4, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="Expected 2D background"):
            create_overlay(bg_3d, [])

    def test_mismatched_shape_raises(self, background):
        wrong_shape = np.ones((8, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            create_overlay(background, [Layer(CHAN_BLUE, wrong_shape)])

    def test_output_shape(self, background, ones_layer):
        layers = [Layer(CHAN_BLUE, ones_layer)]
        result = create_overlay(background, layers)
        assert result.shape == (4, 4, 3)

    def test_output_in_unit_range(self, background, ones_layer):
        layers = [
            Layer(CHAN_BLUE, ones_layer),
            Layer(CHAN_GREEN, ones_layer, opacity=0.5),
        ]
        result = create_overlay(background, layers)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_opacity_preserves_background(self, background, ones_layer):
        result = create_overlay(
            background,
            [Layer(CHAN_BLUE, ones_layer, opacity=0.0)],
        )
        expected = np.full((4, 4, 3), 0.5, dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_alpha_blend_mode(self, background, ones_layer):
        result = create_overlay(
            background,
            [Layer(CHAN_BLUE, ones_layer, blend_mode=BlendMode.ALPHA)],
        )
        assert result.shape == (4, 4, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


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
        result = overlay_channels(
            background,
            {CHAN_BLUE: ones_layer},
            zero_transparent=False,
        )
        assert result.shape == (4, 4, 3)

    def test_alpha_blend_mode(self, background, ones_layer):
        result = overlay_channels(
            background,
            {CHAN_BLUE: ones_layer},
            blend_mode=BlendMode.ALPHA,
        )
        assert result.shape == (4, 4, 3)

    def test_additive_is_default(self, background, ones_layer):
        result = overlay_channels(
            background,
            {CHAN_BLUE: ones_layer},
        )
        assert result.shape == (4, 4, 3)
