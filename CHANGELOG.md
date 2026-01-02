# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-02

### Changed (Breaking)
- Restructured metadata classes with new architecture:
  - `ChannelMetadata` now uses structured components (`PhysicalDimensions`, `AcquisitionSettings`, `MicroscopeSettings`)
  - `ImageMetadata` now contains `sizes` dict and `channel_metadata_list`
  - Added `dimensions` property to derive `DimensionFlags` from all channels
- Changed `Channel` from Enum to dataclass with registration system
  - Now supports custom channels via `Channel.from_wavelength()` and `Channel.from_optical_config_name()`
- Renamed `frame_interval_ms` to `frame_intervals_ms` (now a numpy array of intervals)
- Removed `MULTICHANNEL` flag from individual channel dimensions (now only at image level)

### Added
- New `nikon.py` module with comprehensive ND2 metadata extraction
  - `_NikonMetadataParser` class for parsing Nikon ND2 files
  - Automatic channel detection from optical configuration
  - Frame interval calculation with fallback from duration
  - Support for time-lapse, Z-stack, and RGB dimensions
- New metadata structure classes in `microscopy_utils.py`:
  - `PhysicalDimensions`: height, width, pixel size, z-stack info
  - `AcquisitionSettings`: exposure time, zoom, binning, frame intervals, wavelengths
  - `MicroscopeSettings`: magnification, NA, objective, light source, laser power
  - `DimensionFlags`: bit flags for MULTICHANNEL, TIMELAPSE, Z_STACK, SPECTRAL, RGB, MONTAGE
  - `DimensionValidatorMixin`: validates required fields based on dimension flags
- `MicroscopyImage.from_nd2_path()` class method for loading Nikon ND2 files
- `MicroscopyImage.from_lif_path()` class method for loading Leica LIF files (metadata parsing TODO)
- Convenience properties on `MicroscopyImage`: `sizes`, `dimensions`, `channel_axis`
- Custom `__repr__` for `AcquisitionSettings` that shows mean/std of frame intervals

### Fixed
- `MicroscopyImage.get_intensities_from_channel()` now uses `channel_axis` from metadata instead of assuming axis 0
- Proper handling of None values in microscope settings (magnification, NA)

### Removed
- `batch.py` module and related test modules (`test_batch_generator.py`, `test_image_batch.py`)

## [0.1.0] - 2025-08-06

### Added
- Initial release
- Basic microscopy image processing tools
- Cell/particle segmentation with Cellpose integration
- Support for Nikon ND2 file formats
- Channel management and fluorescence quantification
