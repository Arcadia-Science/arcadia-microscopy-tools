from typing import Any

import numpy as np

from arcadia_microscopy_tools import Channel, MicroscopyImage
from arcadia_microscopy_tools.channels import FITC


def assert_metadata_equal(image: MicroscopyImage, expected_image_metadata: dict[str, Any]):
    for channel_str, known_channel_metadata in expected_image_metadata.items():
        channel = Channel.registry[channel_str]
        channel_index = image.channels.index(channel)
        channel_metadata = image.metadata.image.channel_metadata_list[channel_index]  # type: ignore

        # Iterate through the nested structure (resolution, optics, acquisition)
        for section_name, section_values in known_channel_metadata.items():
            section_obj = getattr(channel_metadata, section_name)

            for parameter_name, known_value in section_values.items():
                parsed_value = getattr(section_obj, parameter_name)
                if isinstance(parsed_value, str):
                    assert parsed_value == known_value
                elif parsed_value is None:
                    # Skip None values
                    continue
                else:
                    assert np.allclose(parsed_value, known_value)


def test_parse_multichannel_metadata(valid_multichannel_nd2_path, known_metadata):
    image = MicroscopyImage.from_nd2_path(valid_multichannel_nd2_path)
    known_image_metadata = known_metadata["example-multichannel.nd2"]
    assert_metadata_equal(image, known_image_metadata)


def test_parse_timelapse_metadata(valid_timelapse_nd2_path, known_metadata):
    known_channels = [FITC]
    image = MicroscopyImage.from_nd2_path(valid_timelapse_nd2_path, channels=known_channels)
    known_image_metadata = known_metadata["example-timelapse.nd2"]
    assert_metadata_equal(image, known_image_metadata)


def test_parse_zstack_metadata(valid_zstack_nd2_path, known_metadata):
    image = MicroscopyImage.from_nd2_path(valid_zstack_nd2_path)
    known_image_metadata = known_metadata["example-zstack.nd2"]
    assert_metadata_equal(image, known_image_metadata)
