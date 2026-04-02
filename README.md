# pinqued_tools
Repository of the package for PINQUED project. 
The package contains classes and utility functions to handle data acquisition, data processing and file reading/writing.

For FLIR camera support, Python 3.10 is required due to compatibility with the Teledyne Spinnaker SDK and its PySpin wrapper. PySpin also requires a NumPy version 1.x, which is reflected in this package's dependencies. For detailed instructions on installing the Spinnaker SDK and PySpin, please refer to the official Teledyne documentation.


# Changelog

## v 0.0.1
-  Still cannot save data to a file (wip)
-  Support for FLIR camera data acquisition
-  Classes for spectral data processing 
-  Classes for spectral data storage