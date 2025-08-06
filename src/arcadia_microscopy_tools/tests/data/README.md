# Test data

## Overview
This directory contains test data files used for validating the microscopy metadata parsing functionality in the `pbmc_diff` package. The files include various types of microscopy images in Nikon's ND2 format along with their known metadata for test verification.

## Files

### Microscopy images

- **example-multichannel.nd2**:
  A multichannel image of PBMC cells, cropped to 256×256 pixels. This image was acquired with brightfield illumination as well as 405 nm, 488 nm, and 561 nm excitation at 20× magnification.
- **example-timelapse.nd2**:
  A ~30 s excerpt from a timelapse of *E. coli* expressing a variant of GFP, cropped to 64×64 pixels. This timelapse was acquired with 488 nm excitation at 40× magnification.
- **example-zstack.nd2**:
  A ~60 µm portion of a z-stack of unknown cells, cropped to 128×128 pixels. This z-stack was acquired with 488 nm excitation at 20× magnification.

### Config files
- **known-metadata.yml**:
  A YAML file that contains known metadata parameters regarding each example image, obtained from reading each file with Nikon's NIS-Elements software.


## Usage
These test files are automatically loaded by pytest fixtures defined in `tests/conftest.py`.


## Data source
The test images were acquired on a Nikon Ti2-E microscope with an ORCA-Fusion BT Digital CMOS camera between March and April 2025. Original images have been cropped to reduce file size while preserving the necessary metadata for testing.

<!-- Future Enhancements -->
<!-- TODO: could include these -->
<!-- <img src="../../assets/example-timelapse.gif" width="128"> -->
<!-- <img src="../../assets/example-zstack.gif" width="128"> -->
