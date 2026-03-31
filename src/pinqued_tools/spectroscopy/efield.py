'''
Contains functions for electric field reconstruction
from Stark-split Rydberg EIT spectra 

Author: Mykhailo Vorobiov
'''
#%%
from typing import Callable, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

def reference_interp(efield: float,
                  reference: tuple[np.ndarray, np.ndarray],
                  n_interp: int = 4, 
                  atol: float = 0.21,
                  return_polynomial: bool = False
                  ):
    '''
    Interpolates between points of the reference for a given E-field.
    Calculates 1st derivative of the reference f(E) dependence.
    '''
    # 1. Unpack samples of the reference E-field and peak positions
    efield_reference, peak_pos_reference = reference

    # 2. Stack a mirrored reference to avoid boundary effects
    mirror_ref_field = -np.flip(efield_reference[1:])
    mirror_peak_pos = np.flip(peak_pos_reference[1:])
    efield_reference = np.concatenate((mirror_ref_field, efield_reference))
    peak_pos_reference = np.concatenate((mirror_peak_pos, peak_pos_reference))

    # 3. Extract a portion of the reference that is closest to the E-field
    #    value `efield` 
    closest_field = np.isclose(efield, efield_reference, atol=atol)
    idx_tmp = np.where(closest_field)
    closest_field_idx = np.min(idx_tmp)
    lower_lim_field = closest_field_idx - n_interp
    upper_lim_field = closest_field_idx + n_interp + 1

    x = efield_reference[lower_lim_field:upper_lim_field]
    y = peak_pos_reference[lower_lim_field :upper_lim_field]

    # 4. Interpolate reference values in-between the known with a 
    #    second degree polynomial and define a polynomial object
    poly_coefs = np.polyfit(x, y, 2)
    polynomial = np.poly1d(poly_coefs)
    
    # 5. Calculate peak position with its 1st and 2nd derivatives wrt E-field
    f = polynomial(efield) # freq. position at `efield`
    df_dE = polynomial.deriv(1)(efield) # first derivative

    if return_polynomial:
        return f, df_dE, polynomial
    return f, df_dE

def read_reference(csv_path: str,
                   amp_rel: list[float]):
    '''
    Reads a file with reference dependence of Stark split positions 
    of Rydberg levels vs. E-field as calculated by ARC or any other 
    method. 
    '''
    # 1. Read refernce values into a dictionary
    arc_ref = pd.read_csv(csv_path).to_dict('list')
    
    # 2. Separate E-field column to future use
    efield = np.array(arc_ref['E-field (V/cm)'])
    
    # 3. Delete E-field column from the original dictionary
    del arc_ref['E-field (V/cm)']

    # 4. Define new dictionary for output
    ref_dict = {'efield': efield}
    
    # 5. Define dictionary with frequency detunings only
    detunings = {}
    for amp, (key, value) in zip(amp_rel, arc_ref.items()):
        detunings[key] = {'freq_detuning': value,
                         'amplitude_relative': amp}
        
    # 6. Add detunings to the `ref_dict` as a dictionary
    ref_dict['stark_components'] = detunings # type: ignore
    return ref_dict

def eit_signal(freq: np.ndarray, # Frequency 
               efield: float, # E-field in units of the reference E-field
               reference_dict: dict,
               params_dict: dict,
               lineshape_func: Callable) -> np.ndarray:
    '''
    Simulate EIT signal for a given electric field
    '''

    # 1. Extract reference E-field and corresponding
    #    frequency detuning samples (claculated separately)
    #    into separate variables.
    efield_ref = reference_dict['efield']
    stark_components_ref = reference_dict['stark_components']

    # 2. Define additional model parameters
    #    `scale_factor` is for overall scaling of the spectrum
    #    `width_0` is the E=0 line width
    #    `gradE_dr` is the gradient of E-field time the distance over pixel.
    scale_factor = params_dict['amp']
    width_0 = params_dict['width_0']
    gradE_dr = params_dict['gradE_dr']

    # 3. Calculate interpolated values for each Stark component 
    #    provided by the reference
    spec_lines_dict = []
    # iterate over Stark components
    for ref in stark_components_ref.values():
        fpos_ref = ref['freq_detuning']
        amp_rel = ref['amplitude_relative']
        # calculate interpolated values
        fpos_tuple = reference_interp(efield,
                                      reference=(efield_ref, fpos_ref))
        fpos, fpos_grad = fpos_tuple # type: ignore
        # calculate line width based on its df/dE and initial width
        width = width_0  - fpos_grad * gradE_dr
        params = {'func': lineshape_func,
                  'width': width, 
                  'fpos': fpos,
                  'amplitude': amp_rel}
        # add resulting dictionary to a list
        spec_lines_dict.append(params)
    
    # 4. Calculate EIT spectrum
    spectrum = scale_factor * simulate_spectrum(freq, spec_lines_dict)
    return spectrum

def fit_spectrum(spectrum_dict: dict):
    '''
    Fits a single spectrum using model defined in `eit_signal()`.
    '''
    freq = spectrum_dict['freq']
    spec = spectrum_dict['specrtrum']
    spec_err = spectrum_dict['spectrum_err']
    params = spectrum_dict['params_init']
    # if list of parameters passed perform retro-fitting
    # including information from the previous results 
    # to regularize and enforce continuity of the reconstruction
    if type(params)==list:
        raise NotImplementedError()
    # otherwise perform standard least squares fitting
     

    # NOTE: PICK UP FROM HERE
    ...



class FieldReference():
    def __init__(self, csv_path: str):
        '''
        Reads a file with reference dependence of Stark split positions 
        of Rydberg levels vs. E-field as calculated by ARC or any other 
        method. 
        '''
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
    def efield(self) -> np.ndarray:
        return self._efield
    
    @property
    def detunings(self) -> Dict[str, np.ndarray]:
        return self._detunings
    
    def interp(self, 
               efield: float, 
               atol: float=0.21, 
               n_interp: int=4) -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        efield_reference = self._efield
        detunings_reference = self._detunings_interpolation_domain
        

        # 3. Extract a portion of the reference that is closest to the E-field
        #    value `efield` 
        closest_field = np.isclose(efield, efield_reference, atol=atol)
        idx_tmp = np.where(closest_field)
        closest_field_idx = np.min(idx_tmp)
        lower_lim_field = closest_field_idx - n_interp
        upper_lim_field = closest_field_idx + n_interp + 1

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


class SignalSimulator():
    def __init__(self, reference: FieldReference):
        pass

    def model(self,):
        pass



#%%
if __name__=='__main__':
    pass
# %%
