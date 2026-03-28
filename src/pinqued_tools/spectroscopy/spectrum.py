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


#%%
default_axis0d_units = {'f': 'MHz'}
default_axis1d_units = {'x': 'mm', 'f': 'MHz'}
default_axis2d_units = {'x': 'mm', 'y': 'mm', 'f': 'MHz'}
default_spectrum_units = {'signal': '%', 'signal_err': '%'}

# -------------------- Axes classes -------------------------

@dataclass 
class BaseAxes(ABC):
    """Abstract base class for all Axes dataclasses."""
    f: np.ndarray # Frequency coordinate

@dataclass
class Axes0D(BaseAxes):
    units: Dict[str, str] = field(default_factory=default_axis0d_units)

@dataclass
class Axes1D(BaseAxes):
    x: np.ndarray # Spatial coordinate
    units: Dict[str, str] = field(default_factory=default_axis1d_units)

@dataclass
class Axes2D(BaseAxes):
    x: np.ndarray # Spatial coordinate x
    y: np.ndarray # Spatial coordinate y
    units: Dict[str, str] = field(default_factory=default_axis2d_units)






# ----------------- Spectral Data classes -----------------
@dataclass
class BaseSpectralData(ABC):
    """Abstract base class for all spectral data types."""
    signal: np.ndarray
    signal_err: np.ndarray
    axes: BaseAxes
    units: Dict[str, str] = field(default_factory=default_spectrum_units)
    metadata: dict|None = None

@dataclass
class Spectrum(BaseSpectralData):
    '''
    Class stores a signle spectrum
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes0D):
            raise TypeError(f"axes must be of type Axes0D, not {type(self.axes)}")

@dataclass
class SpectralStrip(BaseSpectralData):
    '''
    Class stores a single spectrum strip
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes1D):
            raise TypeError(f"axes must be of type Axes1D, not {type(self.axes)}")

@dataclass
class SpectralCube(BaseSpectralData):
    '''
    Class stores a spectral cube
    '''
    def __post_init__(self):
        if not isinstance(self.axes, Axes2D):
            raise TypeError(f"axes must be of type Axes2D, not {type(self.axes)}")




# ------------------ Classes for processing Spectral Data -------------------
class SpectrumProcessor():
    '''
    Class for processing individual spectra 
    '''
    def __init__(self, 
                 spectrum: Spectrum):
        # Store a reference to the original spectrum object to avoid copying
        self._spectrum = spectrum

    @property
    def data(self):
        # Return a copy if modifications are made, or the original if not.
        # For now, returning the original reference.
        return self._spectrum
    
    def remove_baseline(self):
        pass

    def denoise_svd(self):
        pass

class CubePreporcessor():
    '''
    Class for pre-processing a SpectralCube object 
    '''
    def __init__(self,
                 cube: SpectralCube):
        pass

class CubeProcessor():
    '''
    Class for processing a SpectralCube object 
    '''
    def __init__(self,
                 cube: SpectralCube):
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
        return SpectralCube(signal=new_signal, 
                            signal_err=new_signal_err, 
                            units=self._cube.units, 
                            axes=new_axes,
                            metadata=self._cube.metadata)

    def get_spectrum(self, x=None, y=None):
        if x is not None and y is not None:
            # Correctly construct Axes0D for the returned Spectrum
            axes = Axes0D(f=self._cube.axes.f)
            return Spectrum(signal=self._cube.signal[x,y], signal_err=self._cube.signal_err[x,y], axes=axes, units=self._cube.units, metadata=self._cube.metadata)
        
    def get_subcube(self):
        raise NotImplementedError()



#%%
if __name__=='__main__':
    pass
# %%
