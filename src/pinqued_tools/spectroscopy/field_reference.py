
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
            
    @property
    def efield(self) -> NDArray:
        return self._efield
    
    @property
    def detunings(self) -> Dict[str, NDArray]:
        return self._detunings
   
    @property
    def level_labels(self) -> list[str]:
        return list(self._detunings.keys())
    
    def interp(self, efield: float, method='spline') -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        if method == 'poly':
            return self.interp_poly(efield)
        elif method == 'spline':
            return self.interp_spline(efield)
        else:
            raise ValueError("Invalid interpolation method. Choose 'poly' or 'spline'.")

    def interp_poly(self, efield: float) -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        efield_reference = self._efield_interpolation_domain
        detunings_reference = self._detunings_interpolation_domain
        

        # 3. Extract a portion of the reference that is closest to the E-field
        #    value `efield` 
        closest_field = np.isclose(efield, efield_reference, atol=self._atol)
        idx_tmp = np.where(closest_field)
        closest_field_idx = np.min(idx_tmp)
        lower_lim_field = closest_field_idx - self._n_interp
        upper_lim_field = closest_field_idx + self._n_interp + 1

        x = efield_reference[lower_lim_field:upper_lim_field]

        detunings_interpolated = []
        for value in detunings_reference.values():
            y = value[lower_lim_field :upper_lim_field]

        # 4. Interpolate reference values in-between the known values
        #    using a second degree polynomial and define a polynomial object
            poly_coefs = np.polyfit(x, y, 2)
            polynomial = np.poly1d(poly_coefs)

        # 5. Calculate peak position with its 1st and 2nd derivatives wrt E-field
            f = polynomial(efield) # freq. position at `efield`
            df_de = polynomial.deriv(1)(efield) # first derivative
            detunings_interpolated.append((f, df_de))

        return detunings_interpolated

    def interp_spline(self, efield: float) -> list[tuple[float, float]]:
        '''
        Alternative interpolation method using cubic splines.
        '''
        efield_reference = self._efield_interpolation_domain
        detunings_reference = self._detunings_interpolation_domain

        detunings_interpolated = []
        for value in detunings_reference.values():
            cs = CubicSpline(efield_reference, value)
            f = cs(efield) # freq. position at `efield`
            df_de = cs.derivative(1)(efield) # first derivative
            detunings_interpolated.append((f, df_de))

        return detunings_interpolated
    
    def interp_derivative(self, efield: NDArray) -> NDArray:
        detuning_derivatives = {}
        for key in self._detunings.keys():
            cs = CubicSpline(self._efield, self._detunings[key])
            detuning_derivatives[key] = cs.derivative(1)(efield)
        return detuning_derivatives
    
    def interp_values(self, efield: NDArray) -> NDArray:
        detuning_values = {}
        for key in self._detunings.keys():
            cs = CubicSpline(self._efield, self._detunings[key])
            detuning_values[key] = cs(efield)
        return detuning_values