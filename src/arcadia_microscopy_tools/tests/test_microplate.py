import pytest

from arcadia_microscopy_tools.microplate import MicroplateLayout, Well


class TestWell:
    def test_well_creation(self):
        well = Well(id="A01", sample="sample1")
        assert well.id == "A01"
        assert well.sample == "sample1"
        assert well.row == "A"
        assert well.column == 1

    def test_well_id_normalization(self):
        well = Well(id="a1")
        assert well.id == "A01"

    def test_well_invalid_id(self):
        with pytest.raises(ValueError, match="Well ID must be at least 2 characters"):
            Well(id="A")

    def test_well_from_dict(self):
        data = {"well_id": "B02", "sample": "test_sample", "concentration": 10}
        well = Well.from_dict(data)
        assert well.id == "B02"
        assert well.sample == "test_sample"
        assert well.properties["concentration"] == 10


class TestMicroplateLayout:
    def test_layout_creation(self):
        wells = [Well(id="A01", sample="s1"), Well(id="B02", sample="s2")]
        layout = MicroplateLayout(wells)
        assert len(layout) == 2
        assert "A01" in layout
        assert "B02" in layout

    def test_layout_getitem(self):
        wells = [Well(id="A01", sample="s1")]
        layout = MicroplateLayout(wells)
        well = layout["A01"]
        assert well.sample == "s1"

    def test_layout_duplicate_wells(self):
        wells = [Well(id="A01", sample="s1"), Well(id="A01", sample="s2")]
        with pytest.raises(ValueError, match="Duplicate well ID"):
            MicroplateLayout(wells)

    def test_layout_to_dataframe(self):
        wells = [Well(id="A01", sample="s1"), Well(id="B02", sample="s2")]
        layout = MicroplateLayout(wells)
        df = layout.to_dataframe()
        assert len(df) == 2
        assert "well_id" in df.columns
        assert "sample" in df.columns
