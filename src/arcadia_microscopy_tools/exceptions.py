class MetadataWarning(UserWarning):
    """Metadata was incomplete or ambiguous; a fallback value was used."""


class SegmentationWarning(UserWarning):
    """A segmentation step produced a degraded or missing result."""
