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
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import copy

# -------------------- Axes classes -------------------------
@dataclass 
class BaseAxes(ABC):
    """Abstract base class for all Axes dataclasses."""
    f: NDArray # Frequency coordinate

@dataclass
class Axes0D(BaseAxes):
    units: Dict[str, str] = field(default_factory=lambda: {'f': 'MHz'})

@dataclass
class Axes1D(BaseAxes):
    x: NDArray # Spatial coordinate
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'f': 'MHz'})

@dataclass
class Axes2D(BaseAxes):
    x: NDArray # Spatial coordinate x
    y: NDArray # Spatial coordinate y
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'y': 'mm', 'f': 'MHz'})

# ----------------- Spectral Data classes -----------------
@dataclass
class SpectralData():
    '''
    Class stores any set of spectral data
    '''
    signal: NDArray
    axes: BaseAxes
    signal_err: NDArray|None = None
    units: Dict[str, str] = field(default_factory=lambda: {'signal': '%', 'signal_err': '%'})
    metadata: dict|None = None

    def __post_init__(self):
        self.metadata = self.metadata or {'signal dims': f'{self.signal.shape}'}

    def __repr__(self) -> str:
        pstring = f"<{self.__class__.__name__} at {hex(id(self))}>\n"
        total_mem = 0

        pstring += "--- Data Arrays ---\n"
        
        mem_signal = self.signal.nbytes
        total_mem += mem_signal
        pstring += f"signal: shape={self.signal.shape}, unit={self.units.get('signal', 'N/A')}, mem={mem_signal / 1024**2:.3f} MB\n"

        if self.signal_err is not None:
            mem_err = self.signal_err.nbytes
            total_mem += mem_err
            pstring += f"signal_err: shape={self.signal_err.shape}, unit={self.units.get('signal_err', 'N/A')}, mem={mem_err / 1024**2:.3f} MB\n"

        pstring += "\n--- Axes ---\n"
        axes_fields = fields(self.axes)
        for f in axes_fields:
            if f.name == 'units':
                continue
            ax_val = getattr(self.axes, f.name)
            if isinstance(ax_val, np.ndarray):
                mem_ax = ax_val.nbytes
                total_mem += mem_ax
                pstring += f"{f.name}: shape={ax_val.shape}, unit={self.axes.units.get(f.name, 'N/A')}, mem={mem_ax / 1024**2:.3f} MB\n"

        if self.metadata:
            pstring += "\n--- Metadata ---\n"
            for k, v in self.metadata.items():
                pstring += f"{k}: {v}\n"

        pstring += "-------------------\n"
        pstring += f"Total memory: {total_mem / 1024**2:.3f} MB\n"
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
        self._data.signal = self._data.signal.astype(np.float64)

    @property
    def data(self,) -> SpectralData:
        return self._data

    def remove_fmean(self, samples = range(10)) -> None:
        '''
        Removes mean value of the selected samples calculated along the frequency axis.
        '''
        # 1. Extract signal samples to be averaged
        samples = np.take(self._data.signal, samples, axis=0)
        
        # 2. Calculate average and subtract from the signal array
        mean = np.mean(samples, axis=0, keepdims=True)
        mean_full = np.repeat(mean, self._data.signal.shape[0], axis=0)
        self._data.signal -= mean_full

        # 3. Add message to metadata that removal of mean has been performed
        if self._data.metadata is not None:
            self._data.metadata['fmean'] = f'Subtracted mean of f {samples.shape[0]}'
        else:
            self._data.metadata = {'fmean': f'Subtracted mean of f {samples.shape[0]}'}

    def _bin(self, array, px_per_bin: int) -> NDArray:
        '''
        Private fucntion to bin a 1d signal array
        '''
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        return np.mean(arr_reshaped, axis=1)
    
    def _bin_error(self, array, px_per_bin: int) -> NDArray:
        '''
        Private fucntion to calcualte errors for binned 1d signal array
        '''
        # Propagate error when the signal samples are binned
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        # Claculate error of the binned signal as
        #  σ = √ {1/N ∑σ^2 } 
        signal_error = np.sqrt(np.sum(arr_reshaped**2, axis=1) / px_per_bin)
        return signal_error

    def bin(self,
            px_per_bin: int = 2,
            axis: int = 0) -> None:
        '''
        Performs spatial binning of the spectral data.
        So far limited to axis arrays with dimenstions propto powers of 2
        Axes idices:
            0 - f
            1 - x
            2 - y
        '''
        if self._data.signal.shape[axis] % px_per_bin != 0:
            print(f'Cannot perfrom binning along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin.')
            print(f'Make sure axis length is dvisible by {px_per_bin}.')
            raise ValueError

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
    # ------------------  Usage example ----------------
    from datetime import datetime
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    from pinqued_tools.data.io import SpectralDataH5Handler
    h5_handler = SpectralDataH5Handler()

    # Create mock data
    sdata0 = SpectralData(signal=10 + np.random.poisson(lam=100, size=256)*10.1,
                          signal_err=np.sqrt(100 + np.random.poisson(lam=100, size=256)*10.1),
                         axes=Axes0D(f=np.linspace(0,10,256)),
                         metadata={'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    sdata = SpectralData(signal=100 + np.random.poisson(lam=100, size=(256,256))*10.1,
                         axes=Axes1D(x=np.linspace(0,10,256), 
                                     f=np.linspace(0,100,256)))

    # Process data
    sproc = SpectralDataProcessor(sdata0)
    sproc.remove_fmean()
    sproc.bin(px_per_bin=16, axis=0)
    sdata_new = sproc.data

    print(sdata0)
    print(sdata_new)

    h5_handler.save(sdata0, '.hello_h5file.h5')
    sdata0 = h5_handler.load('.hello_h5file.h5')
    print(sdata0)

    # Plot 0D (single spectrum)
    fig, ax = plt.subplots()
    ax.errorbar(x=sdata0.axes.f, y=sdata0.signal, yerr=sdata0.signal_err, 
                linestyle='None', marker='o', markersize=2, alpha=0.6)
    ax.errorbar(x=sdata_new.axes.f, y=sdata_new.signal, yerr=sdata_new.signal_err, 
                linestyle='None', marker='v', markersize=2, alpha=0.6)
    # ax.set_xlabel(f'Detuning $\\Delta_c$ ({sdata.axes.units['f']})')
    # ax.set_ylabel(f'EIT Signal $S$ ({sdata.units['signal']})')

    # Plot 1D (spatial-frequnecy map)
    fig, ax = plt.subplots()
    ax.pcolormesh(sdata.axes.x, sdata.axes.f, sdata.signal, cmap='jet')
    # ax.set_xlabel(f'Position $x$ ({sdata.axes.units['x']})')
    # ax.set_ylabel(f'Detuning $\\Delta_c$ ({sdata.axes.units['f']})')

# %%
