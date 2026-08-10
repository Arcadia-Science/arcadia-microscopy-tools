from dataclasses import FrozenInstanceError, replace

import pytest

from arcadia_microscopy_tools.channels import (
    BRIGHTFIELD,
    CHANNELS,
    DAPI,
    E_CARS,
    F_CARS,
    SRS,
    BrightfieldChannel,
    CARSChannel,
    Channel,
    DetectionGeometry,
    FluorescenceChannel,
    Modality,
    SHGChannel,
    SpectralChannel,
)
from arcadia_microscopy_tools.display import channel_color, wavelength_to_hex


class TestChannelHierarchy:
    def test_base_channel_is_abstract(self):
        with pytest.raises(TypeError):
            Channel()  # type: ignore[abstract]

    def test_spectral_channel_is_abstract(self):
        with pytest.raises(TypeError):
            SpectralChannel()  # type: ignore[abstract]

    def test_fluorescence_is_spectral(self):
        assert isinstance(DAPI, SpectralChannel)

    def test_brightfield_is_not_spectral(self):
        assert not isinstance(BRIGHTFIELD, SpectralChannel)


class TestFluorescenceChannel:
    def test_basic_creation(self):
        ch = FluorescenceChannel(excitation_nm=488, emission_nm=509, label="GFP")
        assert ch.name == "GFP"
        assert ch.excitation_nm == 488
        assert ch.emission_nm == 509
        assert ch.modality is Modality.FLUORESCENCE
        assert ch.geometry is DetectionGeometry.EPI

    def test_default_name_from_bands(self):
        ch = FluorescenceChannel(excitation_nm=488, emission_nm=512)
        assert ch.name == "Ex488/Em512"

    def test_default_name_excitation_only(self):
        assert FluorescenceChannel(excitation_nm=405).name == "Ex405"

    def test_default_name_emission_only(self):
        assert FluorescenceChannel(emission_nm=520).name == "Em520"

    def test_invalid_excitation_raises(self):
        with pytest.raises(ValueError, match="excitation_nm must be positive"):
            FluorescenceChannel(excitation_nm=-10)

    def test_invalid_emission_raises(self):
        with pytest.raises(ValueError, match="emission_nm must be positive"):
            FluorescenceChannel(emission_nm=0)

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            DAPI.excitation_nm = 999  # ty: ignore[invalid-assignment]


class TestIdentity:
    def test_equality_ignores_label(self):
        a = FluorescenceChannel(excitation_nm=405, emission_nm=450, label="DAPI")
        b = FluorescenceChannel(excitation_nm=405, emission_nm=450, label="Hoechst")
        assert a == b

    def test_equality_uses_bands(self):
        a = FluorescenceChannel(excitation_nm=405, emission_nm=450)
        b = FluorescenceChannel(excitation_nm=405, emission_nm=451)
        assert a != b

    def test_different_modalities_not_equal(self):
        assert FluorescenceChannel(excitation_nm=797) != SRS

    def test_hashable_and_label_independent(self):
        a = FluorescenceChannel(excitation_nm=405, emission_nm=450, label="DAPI")
        b = FluorescenceChannel(excitation_nm=405, emission_nm=450)
        assert hash(a) == hash(b)
        assert {a: "value"}[b] == "value"

    def test_label_free_instances_equal(self):
        assert BrightfieldChannel() == BRIGHTFIELD
        assert hash(BrightfieldChannel()) == hash(BRIGHTFIELD)


class TestDetectionGeometry:
    def test_geometry_distinguishes_cars(self):
        assert E_CARS != F_CARS
        assert E_CARS.name == "E-CARS"
        assert F_CARS.name == "F-CARS"

    def test_cars_geometry_values(self):
        assert E_CARS.geometry is DetectionGeometry.EPI
        assert F_CARS.geometry is DetectionGeometry.FORWARD

    def test_shg_names(self):
        assert SHGChannel(geometry=DetectionGeometry.EPI).name == "E-SHG"
        assert SHGChannel(geometry=DetectionGeometry.FORWARD).name == "F-SHG"


class TestLoaderRebuild:
    """Mimics how loaders fill in per-acquisition wavelengths via replace()."""

    def test_replace_preserves_geometry_and_modality(self):
        built = replace(E_CARS, excitation_nm=797.0, emission_nm=648.0)
        assert built.modality is Modality.CARS
        assert built.geometry is DetectionGeometry.EPI
        assert built.excitation_nm == 797.0
        assert built.emission_nm == 648.0
        assert isinstance(built, CARSChannel)

    def test_rebuilt_differs_from_template(self):
        built = replace(SRS, excitation_nm=797.0, emission_nm=797.0)
        assert built != SRS
        assert built.modality is SRS.modality


class TestPredefinedChannels:
    def test_channels_dict_contains_expected(self):
        assert "DAPI" in CHANNELS
        assert "BRIGHTFIELD" in CHANNELS
        assert "E-CARS" in CHANNELS
        assert len(CHANNELS) == 12

    def test_predefined_channel_properties(self):
        assert DAPI.excitation_nm == 405
        assert DAPI.emission_nm == 450
        assert DAPI.modality is Modality.FLUORESCENCE


class TestChannelColor:
    def test_label_free_color(self):
        assert channel_color(BRIGHTFIELD) == "#FFFFFF"

    def test_fluorescence_derives_from_emission(self):
        assert channel_color(DAPI) == wavelength_to_hex(450)

    def test_out_of_visible_falls_back(self):
        # SRS template has no wavelengths -> NIR fallback color.
        assert channel_color(SRS) == "#8B0000"


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
