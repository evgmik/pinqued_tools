
from typing import Dict
from numpy.typing import NDArray

import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

class FieldReference():
    '''
    Class reads Rydberg levels positions vs E-field and interpolate 
    between calculated values for arbitrary field within the valid reference range.
    Additionally calculates gradient df/dE to account for Stark broadening.
    '''
    def __init__(self, 
                 csv_path: str,
                 atol = 0.3,
                 n_interp = 4):
        '''
        Reads a file with reference dependence of Stark split positions 
        of Rydberg levels vs. E-field as calculated by ARC or any other 
        method. 
        '''

        self._atol = atol
        self._n_interp = n_interp

        # 1. Read refernce values into a dictionary
        arc_ref = pd.read_csv(csv_path).to_dict('list')

        # 2. Separate E-field column and create an extended domain for interpolation
        original_efield_array = np.array(arc_ref['E-field (V/cm)'])
        efield_mirrored = -np.flip(original_efield_array[1:])

        self._efield_interpolation_domain = np.concatenate((efield_mirrored, original_efield_array))
        self._efield = np.array(arc_ref['E-field (V/cm)'])

        # 3. Delete E-field column from the original dictionary
        del arc_ref['E-field (V/cm)']

        # 3. Define dictionary with frequency detunings only
        self._detunings = {}
        self._detunings_interpolation_domain = {}
        for key, detuning in arc_ref.items():
            original_detuning = np.array(detuning)
            detuning_mirrored = np.flip(original_detuning[1:])
            detuning_interpolation_domain = np.concatenate((detuning_mirrored, original_detuning))
            self._detunings[key] = original_detuning
            self._detunings_interpolation_domain[key] = detuning_interpolation_domain
            
        # Precompute splines to speed up interpolation
        self._splines = {}
        for key, val in self._detunings_interpolation_domain.items():
            self._splines[key] = CubicSpline(self._efield_interpolation_domain, val)

    @property
    def efield(self) -> NDArray:
        return self._efield
    
    @property
    def detunings(self) -> Dict[str, NDArray]:
        return self._detunings
   
    @property
    def level_labels(self) -> list[str]:
        return list(self._detunings.keys())
    
    def interp(self, efield: float) -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        return self.interp_spline(efield)

    def interp_spline(self, efield: float) -> list[tuple[float, float]]:
        '''
        Alternative interpolation method using cubic splines.
        '''
        efield_reference = self._efield_interpolation_domain
        detunings_reference = self._detunings_interpolation_domain

        detunings_interpolated = []
        for key in detunings_reference.keys():
            cs = self._splines[key]
            f = cs(efield) # freq. position at `efield`
            df_de = cs(efield, nu=1) # first derivative (faster than .derivative(1)())
            detunings_interpolated.append((f, df_de))

        return detunings_interpolated
    
    def interp_dfdE(self, efield: NDArray) -> Dict[str, NDArray]:
        detuning_derivatives = {}
        for key in self._detunings.keys():
            detuning_derivatives[key] = self._splines[key](efield, nu=1)
        return detuning_derivatives
    
    def interp_f(self, efield: NDArray) -> Dict[str, NDArray]:
        detuning_values = {}
        for key in self._detunings.keys():
            detuning_values[key] = self._splines[key](efield)
        return detuning_values