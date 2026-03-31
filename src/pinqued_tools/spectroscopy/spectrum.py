'''
Classses for spectral data storage and processing

Author: Mykhailo Vorobiov
'''
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataclasses import dataclass, field, fields, asdict
from typing import Dict
from abc import ABC, abstractmethod
import copy

from pinqued_tools.spectroscopy.efield import FieldReference

from pinqued_tools.analysis.plotting import *

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
class SpectralData():
    '''
    Class stores any set of spectral data
    '''
    signal: np.ndarray
    axes: BaseAxes
    signal_err: np.ndarray|None = None
    units: Dict[str, str] = field(default_factory=lambda: {'signal': '%', 'signal_err': '%'})
    metadata: dict|None = None
    def __post_init__(self):
        self.metadata = self.metadata or {'signal dims': f'{self.signal.shape}'}

    def __repr__(self) -> str:
        pstring = ''
        for field in fields(self):
            pstring += f'{field.name}: {getattr(self, field.name)}\n'
        return pstring

            
# ------------------ Classes for processing Spectral Data -------------------

class SpectralDataProcessor():
    '''
    Class for processing spectral data.
    Spectral data can be
        0d - signle spectrum
        1d - 1d spatial map (f, x)
        2d - 2d spatial map (f, x, y)
    '''

    AX_ALIAS = ['f', 'x', 'y']

    def __init__(self, data: SpectralData):
        self._data = copy.deepcopy(data)

    @property
    def data(self,) -> SpectralData:
        return self._data

    def remove_fmean(self, samples = range(10)):
        '''
        Removes mean value of the selected samples calculated along the frequency axis.
        '''
        # 1. Extract signal samples to be averaged
        samples = np.take(self._data.signal, samples, axis=0)
        
        # 2. Calculate average and subtract from the signal array
        mean = np.mean(samples, axis=0, keepdims=True)
        mean_full = np.repeat(mean, self._data.signal.shape[-1], axis=0)
        self._data.signal -= mean_full

        # 3. Add message to metadata that removal of mean has been performed
        if self._data.metadata is not None:
            self._data.metadata['fmean'] = f'Subtracted mean of f {samples.shape[0]}'
        else:
            self._data.metadata = {'fmean': f'Subtracted mean of f {samples.shape[0]}'}

    def _bin(self, array, px_per_bin: int):
        '''
        Private fucntion to bin a 1d signal array
        '''
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        return np.mean(arr_reshaped, axis=1)
    
    def _bin_error(self, array, px_per_bin: int):
        '''
        Private fucntion to calcualte errors for binned 1d signal array
        '''
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        return np.sqrt(np.sum(arr_reshaped**2, axis=1))

    def bin(self,
                    px_per_bin: int = 2,
                    axis: int = 0):
        '''
        Performs spatial binning of the spectral data.
        So far limited to axis arrays with dimenstions propto powers of 2
        Axes idices:
            0 - f
            1 - x
            2 - y
        '''
        # 1. Bin signal and signal error if present
        self._data.signal = np.apply_along_axis(self._bin, axis, 
                                                self._data.signal, 
                                                px_per_bin=px_per_bin)
        if self._data.signal_err is not None:
            self._data.signal_err = np.apply_along_axis(self._bin_error, axis, 
                                                    self._data.signal_err, 
                                                    px_per_bin=px_per_bin)
        # 2. Adjust axes accordingly (reduce number of samples)
        ax_dict = asdict(self._data.axes)
        for i,k in enumerate(ax_dict.keys()):
            if i==axis:
                print(k)
                ax_dict[k] = self._bin(ax_dict[k], px_per_bin=px_per_bin)
                # 3. Add message about binning to metadata
                if self._data.metadata is not None:
                    self._data.metadata['binning'] = f'Binning applied along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin'
                else:
                    self._data.metadata = {'binning': f'Binning applied along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin'}
        
        # 4. Update axes
        if isinstance(self._data.axes, Axes0D):
            self._data.axes = Axes0D(**ax_dict)
        elif isinstance(self._data.axes, Axes1D):
            self._data.axes = Axes1D(**ax_dict)
        elif isinstance(self._data.axes, Axes2D):
            self._data.axes = Axes2D(**ax_dict) 

    def remove_baseline(self, **kwargs):
        pass

    def denoise(self):
        pass


#%%
if __name__=='__main__':
    from datetime import datetime

    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()
    sdata0 = SpectralData(signal=10 + np.random.poisson(lam=100, size=256)*10.1,
                          signal_err=np.sqrt(100 + np.random.poisson(lam=100, size=256)*10.1),
                         axes=Axes0D(f=np.linspace(0,10,256)),
                         metadata={'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    sdata = SpectralData(signal=100 + np.random.poisson(lam=100, size=(256,256))*10.1,
                         axes=Axes1D(x=np.linspace(0,10,256), 
                                     f=np.linspace(0,100,256)))

    sproc = SpectralDataProcessor(sdata0)
    sproc.remove_fmean()
    sproc.bin(px_per_bin=2, axis=0)
    sdata_new = sproc.data

    print(sdata_new)

    fig, ax = plt.subplots()
    ax.plot(sdata0.axes.f, sdata0.signal)
    ax.errorbar(x=sdata0.axes.f, y=sdata0.signal, yerr=sdata0.signal_err, linestyle='None')
    ax.plot(sdata_new.axes.f, sdata_new.signal)
    ax.errorbar(x=sdata_new.axes.f, y=sdata_new.signal, yerr=sdata_new.signal_err, linestyle='None')


    fig, ax = plt.subplots()
    ax.pcolormesh(sdata.axes.x, sdata.axes.f, sdata.signal)
    ax.set_xlabel(sdata.axes.units['x'])
    ax.set_ylabel(sdata.axes.units['f'])

# %%
