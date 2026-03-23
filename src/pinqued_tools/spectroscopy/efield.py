'''
Contains functions for electric field reconstruction
from Stark-split Rydberg EIT spectra 

Author: Mykhailo Vorobiov
'''
#%%

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from pinqued_tools.spectroscopy.spectroscopy import simulate_spectrum, holtsmarkian, gaussian

def reference_interp(efield: float,
                  reference: tuple[np.ndarray, np.ndarray],
                  n_interp: int = 4, 
                  atol: float = 0.21,
                  return_polynomial: bool = False
                  ):
    '''
    Interpolates between points of the reference for a given E-field.
    Calculates 1st and 2nd derivatives of the reference.
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
    arc_ref = pd.read_csv(csv_path).to_dict('list')
    efield = np.array(arc_ref['E-field (V/cm)'])
    del arc_ref['E-field (V/cm)']
    ref_dict = {'efield': efield}
    detunings = {}
    for amp, (key, value) in zip(amp_rel, arc_ref.items()):
        detunings[key] = {'freq_detuning': value,
                         'amplitude_relative': amp}
    ref_dict['stark_components'] = detunings
    return ref_dict

def eit_signal(freq: np.ndarray, # Frequency 
               efield: float, # E-field in units of reference E-field
               reference_dict: dict,
               params_dict: dict,
               lineshape_func):
    '''
    Simulate EIT signal for a given electric field
    '''

    efield_ref = reference_dict['efield']
    stark_components_ref = reference_dict['stark_components']

    width_0 = params_dict['width_0']
    gradE_dr = params_dict['gradE_dr']

    spec_lines_dict = []
    for ref in stark_components_ref.values():
        fpos_ref = ref['freq_detuning']
        amp_rel = ref['amplitude_relative']
        fpos_tuple = reference_interp(efield,
                                      reference=(efield_ref, fpos_ref))
        fpos, fpos_grad = fpos_tuple
        print(fpos_tuple)
        print(fpos, fpos_grad)
        width = width_0  - fpos_grad * gradE_dr
        params = {'func': lineshape_func,
                  'width': width, 
                  'fpos': fpos,
                  'amplitude': amp_rel}
        spec_lines_dict.append(params)
        
    spectrum = params_dict['amp'] * simulate_spectrum(freq, spec_lines_dict)
    return spectrum
#%%
if __name__=='__main__':

    amp_rel = [1.35,
               1.35,
               np.sqrt(6),
               np.sqrt(6),
               np.sqrt(6)]
    file_path = 'G:/My Drive/Vaults/WnM-AMO/__Scripts/calculated_stark_maps/stark_map_25D_MHz.csv'
    ref = read_reference(file_path, amp_rel)
    params = {'amp': 4e2,
              'width_0': 20.0,
              'gradE_dr': 10.0}

    freq = np.linspace(100, -1.5e3, 638)
    signal = eit_signal(freq, 
                        efield=25.0,
                        reference_dict=ref,
                        params_dict=params,
                        lineshape_func=holtsmarkian)


    fig, ax = plt.subplots()
    ax.plot(freq, signal)
    ax.set_xlabel('Freq. detuning (MHz)')
    ax.set_ylabel('EIT signal $S$ (arb. units)')
# %%
