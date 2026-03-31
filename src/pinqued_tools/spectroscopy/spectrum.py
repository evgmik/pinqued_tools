'''
Classses for spectral data storage and processing

Author: Mykhailo Vorobiov
'''
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Dict
from abc import ABC, abstractmethod

from pinqued_tools.spectroscopy.efield import FieldReference

#%%
# -------------------- Axes classes -------------------------
@dataclass 
class BaseAxes(ABC):
    """Abstract base class for all Axes dataclasses."""
    f: np.ndarray # Frequency coordinate

@dataclass
class Axes0D(BaseAxes):
    units: Dict[str, str] = field(default_factory=lambda: {'f': 'MHz'})

@dataclass
class Axes1D(BaseAxes):
    x: np.ndarray # Spatial coordinate
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'f': 'MHz'})

@dataclass
class Axes2D(BaseAxes):
    x: np.ndarray # Spatial coordinate x
    y: np.ndarray # Spatial coordinate y
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'y': 'mm', 'f': 'MHz'})

# ----------------- Spectral Data classes -----------------
@dataclass
class BaseSpectralData(ABC):
    """Abstract base class for all spectral data types."""
    signal: np.ndarray
    axes: BaseAxes
    signal_err: np.ndarray|None = None
    units: Dict[str, str] = field(default_factory=lambda: {'signal': '%', 'signal_err': '%'})
    metadata: dict|None = None

@dataclass
class SpectralDataSpec(BaseSpectralData):
    '''
    Class stores a signle spectrum
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes0D):
            raise TypeError(f"axes must be of type Axes0D, not {type(self.axes)}")

@dataclass
class SpectralDataStrip(BaseSpectralData):
    '''
    Class stores a single spectrum strip
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes1D):
            raise TypeError(f"axes must be of type Axes1D, not {type(self.axes)}")

@dataclass
class SpectralDataCube(BaseSpectralData):
    '''
    Class stores a spectral cube
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes2D):
            raise TypeError(f"axes must be of type Axes2D, not {type(self.axes)}")


# ------------------ Classes for processing Spectral Data -------------------
class BaseSpectralDataProcessor(ABC):
    """Abstract base class for spectral data processors."""
    def __init__(self, 
                 data: BaseSpectralData):
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data

    def remove_baseline(self):
        pass

    def denoise_svd(self):
        pass


class SpecProcessor(BaseSpectralDataProcessor):
    '''
    Class for processing individual spectra 
    '''
    def __init__(self, 
                 data: SpectralDataSpec):
        # Store a reference to the original spectrum object to avoid copying
        self._spectrum = data


        self._signal = data.signal
        self._signal_err = data.signal_err
        self._axes = data.axes
        self._units = data.units
        self._metadata = data.metadata


    @property
    def data(self):
        # Return a copy if modifications are made, or the original if not.
        # For now, returning the original reference.
        return self._spectrum
    
    def remove_baseline(self):
        pass

    def denoise_svd(self):
        pass

class StripProcessor(BaseSpectralDataProcessor):
    pass

class CubeProcessor(BaseSpectralDataProcessor):
    '''
    Class for processing a SpectralCube object 
    '''
    def __init__(self,
                 cube: SpectralDataCube):
        # Store a reference to the original cube object to avoid copying
        self._cube = cube

    @property
    def data(self):
        # Return a copy if modifications are made, or the original if not.
        # For now, returning the original reference.
        return self._cube

    def bin_spatial(self,
                    px_per_bin: int = 2,
                    axis: int = 1):
        if axis > 1 or axis < 0:
            raise ValueError("Axis must be 0 (x) or 1 (y) for 2D spatial binning.")
        
        # Create new binned data and axes, rather than modifying in place
        # This example assumes simple slicing for binning, more complex binning
        # would involve averaging/summing.
        new_signal = self._cube.signal[::px_per_bin, :, :] if axis == 0 else self._cube.signal[:, ::px_per_bin, :]
        new_signal_err = self._cube.signal_err[::px_per_bin, :, :] if axis == 0 else self._cube.signal_err[:, ::px_per_bin, :]
        
        new_x = self._cube.axes.x[::px_per_bin] if axis == 0 else self._cube.axes.x
        new_y = self._cube.axes.y[::px_per_bin] if axis == 1 else self._cube.axes.y

        new_axes = Axes2D(x=new_x, y=new_y, f=self._cube.axes.f, units=self._cube.axes.units)
        return SpectralDataCube(signal=new_signal, 
                            signal_err=new_signal_err, 
                            units=self._cube.units, 
                            axes=new_axes,
                            metadata=self._cube.metadata)

    def get_spectrum(self, x=None, y=None):
        if x is not None and y is not None:
            # Correctly construct Axes0D for the returned Spectrum
            axes = Axes0D(f=self._cube.axes.f)
            return SpectralDataSpec(signal=self._cube.signal[x,y], signal_err=self._cube.signal_err[x,y], axes=axes, units=self._cube.units, metadata=self._cube.metadata)
        
    def get_subcube(self):
        raise NotImplementedError()



#%%
if __name__=='__main__':
    pass
# %%
