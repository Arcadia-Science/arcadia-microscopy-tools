from __future__ import annotations

from arcadia_pycolor import HexCode
from matplotlib.colors import LinearSegmentedColormap

from .channels import Channel


def create_opaque_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a colormap from black to the given color."""
    colors = [
        (0, 0, 0, 1),
        color.hex_code,
    ]
    name = color.name if name is None else name
    colormap = LinearSegmentedColormap.from_list(name, colors)
    return colormap


def create_semitransparent_colormap(
    color: HexCode,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create a semi-transparent colormap for the given color."""
    colors = [
        (1, 1, 1, 0),
        color.hex_code,
    ]
    name = color.name if name is None else name
    colormap = LinearSegmentedColormap.from_list(name, colors)
    return colormap


def channel_to_opaque_colormap(channel: Channel) -> LinearSegmentedColormap:
    """Create an opaque colormap from a Channel's color.

    Args:
        channel: The Channel to create a colormap from.

    Returns:
        A LinearSegmentedColormap from black to the channel's color.

    Raises:
        ValueError: If the channel has no color defined.
    """
    if channel.color is None:
        raise ValueError(f"Channel {channel.name} has no color")
    return create_opaque_colormap(color=channel.color, name=channel.name)


def channel_to_semitransparent_colormap(channel: Channel) -> LinearSegmentedColormap:
    """Create a semi-transparent colormap from a Channel's color.

    Args:
        channel: The Channel to create a colormap from.

    Returns:
        A LinearSegmentedColormap from transparent white to the channel's color.

    Raises:
        ValueError: If the channel has no color defined.
    """
    if channel.color is None:
        raise ValueError(f"Channel {channel.name} has no color")
    return create_semitransparent_colormap(color=channel.color, name=channel.name)
