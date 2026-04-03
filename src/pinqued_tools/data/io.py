'''
Docstring for pinqued_tools.data.io

The class handles reading data from disk.
'''

from abc import ABC, abstractmethod

import os
from datetime import datetime
import h5py
import json
from dataclasses import fields
from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes0D, Axes1D, Axes2D


class DataManager:
    """
    Manages reading and writing data by delegating to format-specific handlers.
    Automates the creation of unique, dated, and versioned filenames.
    """
    def __init__(self, base_path: str = './'):
        self._base_path = os.path.abspath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)
        self._handlers = {}
        # Register default handlers upon initialization
        self.register_handler('.h5', SpectralDataH5Handler())
        # Example for the future:
        # self.register_handler('.json', SpectralDataJsonHandler())

    def register_handler(self, extension: str, handler: 'BaseIOHandler'):
        """
        Registers an I/O handler for a given file extension.
        This makes the DataManager extensible to new file types.
        """
        self._handlers[extension.lower()] = handler

    def _get_handler(self, filepath: str) -> 'BaseIOHandler':
        """Finds the appropriate handler based on the file extension."""
        extension = os.path.splitext(filepath)[1].lower()
        if not extension:
            raise ValueError("File path must have an extension to determine the handler.")
        
        handler = self._handlers.get(extension)
        if not handler:
            raise ValueError(f"No handler registered for file extension '{extension}'.")
        return handler

    def _generate_unique_filepath(self, base_name: str, ext: str) -> str:
        """
        Generates a unique filepath in the format:
        <base_path>/<YYYY-MM-DD>_<base_name>_<count>.<ext>
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename_prefix = f"{base_name}_{date_str}"
        
        count = 0
        while True:
            candidate_filename = f"{filename_prefix}_{count}{ext}"
            filepath = os.path.join(self._base_path, candidate_filename)
            if not os.path.exists(filepath):
                return filepath
            count += 1

    def save(self, data, base_name: str, ext: str = '.h5', **kwargs):
        """
        Generates a unique, dated, and versioned filename, and saves data
        using the appropriate handler based on the extension.
        """
        # Generate the full, unique path for the new file
        filepath = self._generate_unique_filepath(base_name, ext)
        
        # Get the handler and save the data
        handler = self._get_handler(filepath)
        print(f"Saving data to {filepath} using {handler.__class__.__name__}")
        handler.save(data, filepath, **kwargs)

    def save_object(self, obj, base_name: str, ext: str, **kwargs):
        """
        Generates a unique, dated, and versioned filename, and saves data
        using the object's own .save(file_path) method.
        """
        if not hasattr(obj, 'save') or not callable(getattr(obj, 'save')):
            raise TypeError("Object must have a callable 'save' method.")
        
        filepath = self._generate_unique_filepath(base_name, ext)
        print(f"Saving object to {filepath}")
        obj.save(filepath, **kwargs)

    def save_figure(self, fig, base_name: str, ext: str = '.png', **kwargs):
        """
        Generates a unique, dated, and versioned filename, and saves a figure
        using the appropriate handler based on the extension.
        """
        filepath = self._generate_unique_filepath(base_name, ext)
        print(f"Saving object to {filepath}")
        fig.savefig(filepath, **kwargs)

    def load(self, filepath: str, **kwargs):
        """
        Loads data from a file by finding the correct handler and delegating
        the load operation to it. Expects a full or relative path to the file.
        """
        handler = self._get_handler(filepath)
        # Assumes filepath is relative to the current working dir or an absolute path
        print(f"Loading data from {filepath} using {handler.__class__.__name__}")
        return handler.load(filepath, **kwargs)


class BaseIOHandler(ABC):
    """
    Abstract base class for all I/O handlers. Defines the interface
    that DataManager uses to interact with different file formats.
    """
    @abstractmethod
    def save(self, data, file_path: str, **kwargs):
        """Abstract method to save data to a file."""
        pass

    @abstractmethod
    def load(self, file_path: str, **kwargs):
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