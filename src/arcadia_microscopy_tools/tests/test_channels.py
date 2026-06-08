from dataclasses import FrozenInstanceError

import pytest

from arcadia_microscopy_tools.channels import (
    CHANNELS,
    DAPI,
    Channel,
    wavelength_to_hex,
)


class TestChannel:
    def test_basic_creation(self):
        ch = Channel("GFP", "#00FF00", excitation_nm=488, emission_nm=509)
        assert ch.name == "GFP"
        assert ch.color == "#00FF00"
        assert ch.excitation_nm == 488
        assert ch.emission_nm == 509

    def test_color_only(self):
        ch = Channel("BF", "#FFFFFF")
        assert ch.excitation_nm is None
        assert ch.emission_nm is None

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="hex code"):
            Channel("Bad", "not-a-color")

    def test_invalid_excitation_raises(self):
        with pytest.raises(ValueError, match="excitation_nm must be positive"):
            Channel("Bad", "#FF0000", excitation_nm=-10)

    def test_invalid_emission_raises(self):
        with pytest.raises(ValueError, match="emission_nm must be positive"):
            Channel("Bad", "#FF0000", emission_nm=0)

    def test_frozen(self):
        ch = Channel("Frozen", "#AABBCC")
        with pytest.raises(FrozenInstanceError):
            ch.name = "Changed"  # type: ignore

    def test_equality(self):
        a = Channel("X", "#111111", excitation_nm=405)
        b = Channel("X", "#111111", excitation_nm=405)
        assert a == b

    def test_hashable(self):
        d = {DAPI: "value"}
        assert d[DAPI] == "value"


class TestFromWavelength:
    def test_excitation_default(self):
        ch = Channel.from_wavelength(488)
        assert ch.name == "488nm"
        assert ch.excitation_nm == 488
        assert ch.emission_nm is None
        assert ch.color.startswith("#")

    def test_emission(self):
        ch = Channel.from_wavelength(520, is_excitation=False)
        assert ch.emission_nm == 520
        assert ch.excitation_nm is None

    def test_custom_name(self):
        ch = Channel.from_wavelength(488, name="GFP")
        assert ch.name == "GFP"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="360.*780"):
            Channel.from_wavelength(200)
        with pytest.raises(ValueError, match="360.*780"):
            Channel.from_wavelength(1000)


class TestWavelengthToHex:
    def test_returns_valid_hex(self):
        h = wavelength_to_hex(500)
        assert h.startswith("#")
        assert len(h) == 7

    def test_boundaries(self):
        assert wavelength_to_hex(360).startswith("#")
        assert wavelength_to_hex(780).startswith("#")

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            wavelength_to_hex(350)
        with pytest.raises(ValueError):
            wavelength_to_hex(800)


class TestPredefinedChannels:
    def test_channels_dict_contains_all(self):
        assert "DAPI" in CHANNELS
        assert "BRIGHTFIELD" in CHANNELS
        assert "CY5" in CHANNELS
        assert len(CHANNELS) == 12

    def test_predefined_channel_properties(self):
        assert DAPI.excitation_nm == 405
        assert DAPI.emission_nm == 450
        assert DAPI.color == "#0033FF"
