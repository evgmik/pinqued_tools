'''
Docstring for pinqued_tools.data.io

The class handles reading data from disk.
'''

from abc import ABC, abstractmethod

import h5py
import json
from dataclasses import fields
from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes0D, Axes1D, Axes2D


class BaseIOHandler(ABC):
    pass

    @abstractmethod
    def save(self, data, file_path: str):
        """Abstract method to save data to a file."""
        pass

    @abstractmethod
    def load(self, file_path: str):
        """Abstract method to load data from a file."""
        pass

class SpectralDataH5Handler(BaseIOHandler):
    '''
    Class for saving and loading SpectralData objects to/from HDF5 files.
    '''
    def save(self, 
             data: SpectralData, 
             file_path: str, 
             group_name: str = 'spectral_data',
             compression='gzip',
             compression_opts=9):
        """
        Saves a SpectralData object to an HDF5 file.
            TODO: If Spectral data is if type float64, find a way to store as int16 type (save space)
        """

        with h5py.File(file_path, 'a') as f:
            group = f.create_group(group_name)

            # Save arrays and attributes
            group.create_dataset('signal', 
                                 data=data.signal,
                                 compression=compression,
                                 compression_opts=compression_opts)
            if data.signal_err is not None:
                group.create_dataset('signal_err', 
                                     data=data.signal_err,
                                     compression=compression,
                                     compression_opts=compression_opts)
            
            group.attrs['units'] = json.dumps(data.units)
            if data.metadata:
                group.attrs['metadata'] = json.dumps(data.metadata)

            # Save axes
            axes_group = group.create_group('axes')
            axes_group.attrs['axes_type'] = data.axes.__class__.__name__
            axes_group.attrs['units'] = json.dumps(data.axes.units)
            for field in fields(data.axes):
                if field.name != 'units':
                    axes_group.create_dataset(field.name, data=getattr(data.axes, field.name))

    def load(self, file_path: str, group_name: str = 'spectral_data') -> SpectralData:
        """Loads a SpectralData object from an HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            group = f[group_name]
            
            signal = group['signal'][:]
            signal_err = group['signal_err'][:] if 'signal_err' in group else None
            
            units = json.loads(group.attrs['units'])
            metadata = json.loads(group.attrs['metadata']) if 'metadata' in group.attrs else None

            # Load axes
            axes_group = group['axes']
            axes_type_str = axes_group.attrs['axes_type']
            axes_units = json.loads(axes_group.attrs['units'])
            
            axes_data = {name: axes_group[name][:] for name in axes_group}
            axes_data['units'] = axes_units

            axes_class_map = {'Axes0D': Axes0D, 'Axes1D': Axes1D, 'Axes2D': Axes2D}
            axes = axes_class_map[axes_type_str](**axes_data)

            data = SpectralData(signal=signal, 
                                signal_err=signal_err, 
                                axes=axes, 
                                units=units, 
                                metadata=metadata)
            return data