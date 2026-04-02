# pinqued_tools
This repository hosts Python toolkit for the PINQUED project. The package provides a comprehensive set of tools to handle data acquisition, processing, and file I/O.



## Installation

> For FLIR camera support, Python 3.10 is required due to compatibility with the Teledyne Spinnaker SDK and its PySpin wrapper. PySpin also requires a NumPy version 1.x, which is reflected in this package's dependencies. For detailed instructions on installing the Spinnaker SDK and PySpin, please refer to the official Teledyne documentation.

To install PINQUED_tools package, type
```bash
pip install git+https://github.com/mikevorobiov/pinqued_tools/
```



# Changelog
## v0.0.3 - (2026-04-02)
#### Added 
- `DataManager` class that handles saving data and incrementing file name if file path exists.
#### Fixed
- Conversion of `signal` field to float16 in `SpectralDataProcessor` replaced with float32 as hdf5 doesn't support float16.

## v0.0.2 - (2026-04-02)
#### Added
- Implemented an HDF5-based I/O handler (`SpectralDataH5Handler`) to save and load spectral data.
- Introduced a base class for I/O handlers to support more file formats in the future.
- Added a `preprocess` method to the `SpectralDataProcessor` class.

#### Changed
- Updated and improved example usage scripts.

## v0.0.1 - (2026-04-02)
#### Added
- Initial classes for spectral data storage (`SpectralData`) and processing (`SpectralDataProcessor`).
- Support for data acquisition from FLIR cameras, including timestamped frames.
- Implemented single spectrum fitting capabilities.
- Added metadata support to data classes.

#### Changed
- Consolidated multiple spectral data classes into the single `SpectralData` class for a more unified structure.

#### Fixed
- Corrected an error propagation bug that occurred during data binning.
- Addressed a type casting bug in the processing pipeline.
- Improved performance to resolve slow data acquisition with cameras.
- Added a division check to prevent errors when binning data.