from dataclasses import FrozenInstanceError

import pytest
from arcadia_pycolor import HexCode

from arcadia_microscopy_tools.channels import (
    BRIGHTFIELD,
    DAPI,
    FITC,
    Channel,
)


class TestChannelCreation:
    def test_create_basic_channel(self):
        channel = Channel(name="TestChannel")
        assert channel.name == "TestChannel"
        assert channel.excitation_nm is None
        assert channel.emission_nm is None
        assert channel.color is None

    def test_create_channel_with_wavelengths(self):
        channel = Channel(name="GFP", excitation_nm=488, emission_nm=509)
        assert channel.name == "GFP"
        assert channel.excitation_nm == 488
        assert channel.emission_nm == 509

    def test_create_channel_with_color(self):
        color = HexCode("blue", "#0000ff")
        channel = Channel(name="BlueChannel", color=color)
        assert channel.name == "BlueChannel"
        assert channel.color == color

    def test_invalid_excitation_wavelength(self):
        with pytest.raises(ValueError, match="Excitation wavelength must be positive"):
            Channel(name="Bad", excitation_nm=-10)

    def test_invalid_emission_wavelength(self):
        with pytest.raises(ValueError, match="Emission wavelength must be positive"):
            Channel(name="Bad", emission_nm=0)

    def test_channel_is_frozen(self):
        channel = Channel(name="Frozen")
        with pytest.raises(FrozenInstanceError):
            channel.name = "Changed"  # type: ignore


class TestChannelRegistry:
    def test_register_channel(self):
        custom = Channel(name="CUSTOM", excitation_nm=500, emission_nm=520)
        registered = Channel.register(custom)
        assert registered == custom
        assert Channel.get("CUSTOM") == custom

    def test_get_existing_channel(self):
        assert Channel.get("DAPI") == DAPI
        assert Channel.get("dapi") == DAPI  # Case-insensitive
        assert Channel.get("DaPi") == DAPI

    def test_get_nonexistent_channel(self):
        assert Channel.get("NONEXISTENT") is None


class TestFromOpticalConfigName:
    def test_exact_match(self):
        channel = Channel.from_optical_config_name("DAPI")
        assert channel == DAPI

    def test_exact_match_case_insensitive(self):
        channel = Channel.from_optical_config_name("fitc")
        assert channel == FITC

    def test_mono_alias(self):
        channel = Channel.from_optical_config_name("mono")
        assert channel == BRIGHTFIELD

    def test_partial_match(self):
        channel = Channel.from_optical_config_name("DAPI 405nm")
        assert channel == DAPI

    def test_longest_partial_match(self):
        # Register a longer channel name for testing
        fitc_a = Channel(name="FITC-A")
        Channel.register(fitc_a)

        # Should match FITC-A (longer) instead of FITC
        channel = Channel.from_optical_config_name("FITC-A Filter")
        assert channel == fitc_a

    def test_unknown_config_raises_error(self):
        with pytest.raises(ValueError, match="not a known optical configuration"):
            Channel.from_optical_config_name("COMPLETELY_UNKNOWN_12345")


class TestFromEmissionWavelength:
    def test_from_emission_wavelength_basic(self):
        channel = Channel.from_emission_wavelength(520)
        assert channel.name == "520nm"
        assert channel.emission_nm == 520
        assert channel.color is not None
        assert channel.excitation_nm is None

    def test_from_emission_wavelength_with_name(self):
        channel = Channel.from_emission_wavelength(520, name="Green")
        assert channel.name == "Green"
        assert channel.emission_nm == 520

    def test_from_emission_wavelength_with_excitation(self):
        channel = Channel.from_emission_wavelength(520, excitation_nm=488)
        assert channel.excitation_nm == 488
        assert channel.emission_nm == 520

    def test_from_emission_wavelength_invalid_range(self):
        with pytest.raises(ValueError, match="360 and 780 nm"):
            Channel.from_emission_wavelength(200)  # Too low

        with pytest.raises(ValueError, match="360 and 780 nm"):
            Channel.from_emission_wavelength(1000)  # Too high


class TestWavelengthToColor:
    def test_wavelength_to_color_valid(self):
        color = Channel.wavelength_to_color(450)
        assert isinstance(color, HexCode)
        assert color.name == "450nm"
        assert color.hex_code.startswith("#")
        assert len(color.hex_code) == 7

    def test_wavelength_to_color_with_name(self):
        color = Channel.wavelength_to_color(520, name="CustomGreen")
        assert color.name == "CustomGreen"

    def test_wavelength_to_color_boundaries(self):
        # Should work at boundaries
        color_min = Channel.wavelength_to_color(360)
        assert color_min is not None

        color_max = Channel.wavelength_to_color(780)
        assert color_max is not None

    def test_wavelength_to_color_invalid_low(self):
        with pytest.raises(ValueError, match="360 and 780 nm"):
            Channel.wavelength_to_color(350)

    def test_wavelength_to_color_invalid_high(self):
        with pytest.raises(ValueError, match="360 and 780 nm"):
            Channel.wavelength_to_color(800)

    def test_wavelength_to_color_hex_format(self):
        color = Channel.wavelength_to_color(500)
        # Verify hex format
        assert color.hex_code[0] == "#"
        assert all(c in "0123456789ABCDEF" for c in color.hex_code[1:])
