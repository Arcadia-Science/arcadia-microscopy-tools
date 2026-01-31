# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-01-30

### Added
- New `apply_threshold()` image operation for binarizing images
- `preserve_dtype` parameter to Pipeline classes for more flexible control of managing data types during processing

### Changed
- Refactored typing module:
  - Added `UbyteArray` as a type of `ScalarArray`
  - Renamed `FloatArray` to `Float64Array` for clarity

## [0.2.3] - 2026-01-22

### Added
- New `microplate.py` module for managing multiwell plate layouts:
  - `Well` class: Represents individual wells with ID normalization (e.g., "a1" → "A01"), sample tracking, and custom properties
  - `MicroplateLayout` class: Manages complete plate layouts with features:
    - Load layouts from CSV files with `from_csv()`
    - Display plate layouts as formatted grid tables with `display()`

### Changed
- Refactored `masks.py` architecture for improved performance and maintainability:
  - Refactored `MaskProcessor` class into standalone `_process_mask()` function
  - Updated `DEFAULT_CELL_PROPERTY_NAMES` to include circularity and volume by default
- Made cellpose and modal optional dependencies to reduce installation size:
  - Install with `uv pip install arcadia-microscopy-tools[segmentation]` for cellpose support
  - Install with `uv pip install arcadia-microscopy-tools[all]` for all optional dependencies
- Moved pytest from main dependencies to dev group
- Consolidated all dev tools into single `dev` dependency group

### Fixed
- Coordinate format inconsistency in outline extractors: both cellpose and skimage now return outlines in (y, x) format
- `isinstance` check in `SegmentationMask` now accepts `Mapping` type instead of just `dict` to match type annotation
- Empty outline arrays now properly shaped as (0, 2) for consistency

## [0.2.2] - 2026-01-08

### Added
- New `fluorescence_overlays.ipynb` example notebook demonstrating:
  - Sequential blending of fluorescence channels onto brightfield images
- Enhanced blending module with new high-level API:
  - `Layer` dataclass for managing individual fluorescence layers with customizable opacity and transparency
  - `overlay_channels()` function for easy multi-channel overlays with uniform settings
  - `create_sequential_overlay()` function for fine-grained control over individual layers

## [0.2.1] - 2026-01-08

### Added
- New `cell_segmentation.ipynb` example notebook demonstrating:
  - Cell segmentation workflow using Cellpose integration
  - Image preprocessing with percentile-based rescaling
  - Visualization of segmentation results with cell outlines
  - Comprehensive documentation and explanations

## [0.2.0] - 2026-01-02

### Changed (Breaking)
- Restructured metadata classes with new architecture:
  - `ChannelMetadata` now uses structured components (`PhysicalDimensions`, `AcquisitionSettings`, `MicroscopeSettings`)
  - `ImageMetadata` now contains `sizes` dict and `channel_metadata_list`
  - Added `dimensions` property to derive `DimensionFlags` from all channels
- Changed `Channel` from Enum to dataclass with registration system
  - Now supports custom channels via `Channel.from_emission_wavelength()`, `Channel.from_excitation_wavelength()`, and `Channel.from_optical_config_name()`
- Renamed `frame_interval_ms` to `frame_intervals_ms` (now a numpy array of intervals)
- Removed `MULTICHANNEL` flag from individual channel dimensions (now only at image level)

### Added
- New `nikon.py` module with comprehensive ND2 metadata extraction
  - `_NikonMetadataParser` class for parsing Nikon ND2 files
  - Automatic channel detection from optical configuration
  - Frame interval calculation with fallback from duration
  - Support for time-lapse, Z-stack, and RGB dimensions
- New metadata structure classes in `metadata_structures.py`:
  - `PhysicalDimensions`: height, width, pixel size, z-stack info
  - `AcquisitionSettings`: exposure time, zoom, binning, frame intervals, wavelengths
  - `MicroscopeSettings`: magnification, NA, objective, light source, laser power
  - `DimensionFlags`: bit flags for MULTICHANNEL, TIMELAPSE, Z_STACK, SPECTRAL, RGB, MONTAGE
  - `DimensionValidatorMixin`: validates required fields based on dimension flags
- `MicroscopyImage.from_nd2_path()` class method for loading Nikon ND2 files
- `MicroscopyImage.from_lif_path()` class method for loading Leica LIF files (metadata parsing TODO)
- Convenience properties on `MicroscopyImage`: `sizes`, `dimensions`, `channel_axis`

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
