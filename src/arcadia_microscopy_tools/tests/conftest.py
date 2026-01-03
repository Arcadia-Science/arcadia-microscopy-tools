"""
More info regarding these fixtures is provided in
src/arcadia_microscopy_tools/tests/data/README.md
"""

from pathlib import Path

import pytest
import yaml

from arcadia_microscopy_tools import MicroscopyImage


@pytest.fixture
def test_data_directory():
    return Path(__file__).parent / "data"


@pytest.fixture
def valid_multichannel_nd2_path(test_data_directory):
    return test_data_directory / "example-multichannel.nd2"


@pytest.fixture
def valid_timelapse_nd2_path(test_data_directory):
    return test_data_directory / "example-timelapse.nd2"


@pytest.fixture
def valid_zstack_nd2_path(test_data_directory):
    return test_data_directory / "example-zstack.nd2"


@pytest.fixture
def known_metadata(test_data_directory):
    yaml_path = test_data_directory / "known-metadata.yml"
    with yaml_path.open("r") as file:
        parameters = yaml.safe_load(file)
    return parameters


@pytest.fixture
def multichannel_image(valid_multichannel_nd2_path):
    """Create a single multichannel image for testing."""
    return MicroscopyImage.from_nd2_path(valid_multichannel_nd2_path)


@pytest.fixture
def timelapse_image(valid_timelapse_nd2_path):
    """Create a single timelapse image for testing."""
    return MicroscopyImage.from_nd2_path(valid_timelapse_nd2_path)


@pytest.fixture
def zstack_image(valid_zstack_nd2_path):
    """Create a single zstack image for testing."""
    return MicroscopyImage.from_nd2_path(valid_zstack_nd2_path)
