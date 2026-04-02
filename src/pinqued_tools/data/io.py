'''
Docstring for pinqued_tools.data.io

The class handles reading data from disk.
'''

import h5py
import json
from dataclasses import fields
from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes0D, Axes1D, Axes2D

class SpectralDataH5Handler:
    @staticmethod
    def save(self, data: SpectralData, file_path: str, group_name: str = 'spectral_data'):
        """Saves a SpectralData object to an HDF5 file."""
        with h5py.File(file_path, 'a') as f:
            group = f.create_group(group_name)

            # Save arrays and attributes
            group.create_dataset('signal', data=data.signal)
            if data.signal_err is not None:
                group.create_dataset('signal_err', data=data.signal_err)
            
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
    @staticmethod
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

            return SpectralData(signal=signal, signal_err=signal_err, axes=axes, units=units, metadata=metadata)