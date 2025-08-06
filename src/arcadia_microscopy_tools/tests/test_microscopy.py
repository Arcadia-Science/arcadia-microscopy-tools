from typing import Any

import numpy as np

from arcadia_microscopy_tools import Channel, MicroscopyImage


def assert_metadata_equal(image: MicroscopyImage, expected_image_metadata: dict[str, Any]):
    for channel_str, known_channel_metadata in expected_image_metadata.items():
        channel = Channel[channel_str]
        channel_index = image.channels.index(channel)
        channel_metadata = image.metadata.image.channels[channel_index]

        for parameter_name, known_value in known_channel_metadata.items():
            parsed_value = getattr(channel_metadata, parameter_name)
            if isinstance(parsed_value, str):
                assert parsed_value == known_value
            else:
                assert np.allclose(parsed_value, known_value)


def test_parse_multichannel_metadata(valid_multichannel_nd2_path, known_metadata):
    image = MicroscopyImage.from_nd2_path(valid_multichannel_nd2_path)
    known_image_metadata = known_metadata["example-multichannel.nd2"]
    assert_metadata_equal(image, known_image_metadata)


def test_parse_timelapse_metadata(valid_timelapse_nd2_path, known_metadata):
    image = MicroscopyImage.from_nd2_path(valid_timelapse_nd2_path)
    known_image_metadata = known_metadata["example-timelapse.nd2"]
    assert_metadata_equal(image, known_image_metadata)


def test_parse_zstack_metadata(valid_zstack_nd2_path, known_metadata):
    image = MicroscopyImage.from_nd2_path(valid_zstack_nd2_path)
    known_image_metadata = known_metadata["example-zstack.nd2"]
    assert_metadata_equal(image, known_image_metadata)
