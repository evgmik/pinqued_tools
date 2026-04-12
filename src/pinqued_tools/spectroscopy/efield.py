'''
Contains classes for electric field reconstruction
from Stark-split Rydberg EIT spectra 

Author: Mykhailo Vorobiov
'''
#%%
from typing import Callable, Dict
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from scipy.interpolate import CubicSpline
from lmfit import minimize, Parameters, fit_report

from arc import Rubidium85

from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes0D
from pinqued_tools.spectroscopy.lineshapes import HoltsmarkLine

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

    def get_relative_intensities(self, 
                                 init_state: tuple[float, float, float],
                                 final_state: tuple[float, float, float]
                                 ) -> list[float]:

        atom = Rubidium85()

        # 1. Define the states
        n_int, l_int, j_int = init_state   # Intermediate 5P_3/2
        n_ryd, l_ryd, j_ryd = final_state # Target 25D_5/2

        # 2. Dictionary to hold the total intensity for each degenerate |m_j| pair
        # For D_5/2, the allowed |m_j| values are 1/2, 3/2, 5/2
        intensities = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0}

        # 3. Sum the transition strengths
        # Perpendicular polarization is a superposition of q = +1 and q = -1
        for q in [-1, 1]:
            # Loop over all possible initial m_j states in the 5P_3/2 level
            for mj_int in [-1.5, -0.5, 0.5, 1.5]:

                # Calculate the resulting final m_j based on the selection rule
                mj_ryd = mj_int + q

                # Check if this final m_j actually exists in the J=5/2 state
                if abs(mj_ryd) <= j_ryd:

                    # Get the dipole matrix element (in units of ea_0)
                    dipole = atom.getDipoleMatrixElement(n_int, l_int, j_int, mj_int, 
                                                         n_ryd, l_ryd, j_ryd, mj_ryd, q)

                    # The signal intensity is proportional to the square of the dipole element
                    line_strength = abs(dipole)**2

                    # Add it to the corresponding |m_j| component (since +/- m_j are degenerate)
                    intensities[abs(mj_ryd)] += line_strength

        # 4. Normalize the intensities relative to the strongest peak
        max_intensity = max(intensities.values())
        for mj in intensities:
            intensities[mj] /= max_intensity

        # 5. Output the results
        print("Relative Intensities of 25D_5/2 Stark Components")
        print("(Perpendicular Polarization, Delta m_j = +/- 1):")
        print("-" * 50)
        print(f"|m_j| = 1/2 peak: {intensities[0.5]:.3f}")
        print(f"|m_j| = 3/2 peak: {intensities[1.5]:.3f}")
        print(f"|m_j| = 5/2 peak: {intensities[2.5]:.3f}")

        return [intensities[0.5], intensities[1.5], intensities[2.5]]

    def get_relative_intensities_intermediate(self,) -> list[float]:

        atom = Rubidium85()
        # 1. Define the three levels of the EIT ladder
        n_g, l_g, j_g = 5, 0, 0.5   # Initial: 5S_1/2
        n_i, l_i, j_i = 5, 1, 1.5   # Intermediate: 5P_3/2
        n_r, l_r, j_r = 25, 2, 2.5  # Final: 25D_5/2

        # Dictionary to hold the total two-photon intensity for each degenerate |m_j| pair
        intensities = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0}

        # 2. Helper function to calculate the dipole moment for linearly x-polarized light
        def d_x(n1, l1, j1, mj1, n2, l2, j2, mj2):
            """Calculates <2 | d_x | 1> using the spherical tensor components q = +/- 1"""
            d_minus = atom.getDipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, -1)
            d_plus  = atom.getDipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, 1)
            return (d_minus - d_plus) / np.sqrt(2)

        # 3. Sum over all possible initial ground states (unpolarized thermal gas)
        for mj_g in [-0.5, 0.5]:

            # Calculate the transition to every possible final m_j state in the Rydberg level
            for mj_r in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:

                # Initialize the coherent two-photon amplitude for this specific Initial -> Final path
                M_2photon = 0.0

                # Coherently sum the amplitudes over all possible intermediate 5P_3/2 states
                for mj_i in [-1.5, -0.5, 0.5, 1.5]:

                    # Amplitude for 780nm probe (Ground -> Intermediate)
                    d1 = d_x(n_g, l_g, j_g, mj_g, n_i, l_i, j_i, mj_i)

                    # Amplitude for 480nm coupling (Intermediate -> Rydberg)
                    d2 = d_x(n_i, l_i, j_i, mj_i, n_r, l_r, j_r, mj_r)

                    # Multiply the step amplitudes and add to the total coherent path
                    M_2photon += (np.abs(d1) * np.abs(d2)) **2
                # The true EIT line strength is the square of the total coherent amplitude
                line_strength = M_2photon

                # Add the intensity to the corresponding |m_j| component (since +/- m_j are degenerate)
                abs_mj = round(abs(mj_r), 1)
                intensities[abs_mj] += line_strength

        # 4. Normalize the intensities relative to the strongest peak for easy fitting
        max_intensity = max(intensities.values())
        for mj in intensities:
            intensities[mj] /= max_intensity

        # 5. Output the results
        print("Relative Two-Photon EIT Intensities (5S_1/2 -> 5P_3/2 -> 25D_5/2)")
        print("Lasers: Both linearly polarized perpendicular to the DC field (x-axis)")
        print("-" * 70)
        print(f"|m_j| = 1/2 peak: {intensities[0.5]:.3f}")
        print(f"|m_j| = 3/2 peak: {intensities[1.5]:.3f}")
        print(f"|m_j| = 5/2 peak: {intensities[2.5]:.3f}")

        return [intensities[0.5], intensities[1.5], intensities[2.5]]

class SignalSimulator():
    '''
    Class that based on the Rydberg levels Stark splitting 
    from generated by the `FieldReference` class simulates EIT spectra.
    '''
    def __init__(self, 
                 reference: FieldReference,
                 lineshape_func: Callable|None = None
                 ):
        self._reference = reference
        self._lineshape_func = lineshape_func

        if lineshape_func is None:
            print('No lineshape function provided, using Holtsmark lineshape by default.')
            self._hline_list = self._holtsmark_spectrum_prepare()
            print(len(self._hline_list))


    def _holtsmark_spectrum_prepare(self) -> list[HoltsmarkLine]:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        line_keys = self._reference.level_labels
        efield_reference = self._reference.efield
        stark_reference = self._reference.detunings[line_keys[0]]
        
        hline_list = []
        for key in line_keys:
             stark_reference = self._reference.detunings[key]
             hline = HoltsmarkLine(efield_reference, stark_reference)
             hline_list.append(hline)
        print(f'Ready to simulate spectra with Holtsmark spectrum! Number of lines {len(hline_list)}')
        return hline_list

    
    def holtsmark_spectrum(self, 
                           freq: NDArray, 
                           params: Parameters
                           ) -> NDArray:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        efield = params['efield'].value
        scale_factor = params['amp'].value
        width = params['width'].value
        E0 = params['E0'].value
        r_amp = [params[f'rel_amp_{i}'].value for i in range(len(self._hline_list))]
        spectrum = np.zeros_like(freq)
        for hline, ai in zip(self._hline_list, r_amp):
            spectrum += hline(freq, efield, width, E0, ai)
        spectrum *= scale_factor
        return spectrum

    def holtsmark_spectrum_bg(self, 
                           freq: NDArray, 
                           params: Parameters
                           ) -> NDArray:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        signal = self.holtsmark_spectrum(freq, params)
        bg = self.bg_drifts(freq, params, poly_terms=2)
        spectrum = signal + bg
        return spectrum

    def signal(self, 
               freq: NDArray, 
               params: Parameters,
               **kwargs) -> NDArray[np.float64]:
        '''
        Simulate EIT signal for a given electric field
        '''
        scale_factor = params['amp'].value
        width_0 = params['width_0'].value
        gradE_dr = params['gradE_dr'].value
        efield = params['efield'].value

        ref = self._reference.interp(efield)

        spectrum = np.zeros_like(freq)
        r_amp = [params[f'rel_amp_{i}'].value for i in range(len(ref))]
        for (fpos, df_de), amp in zip(ref, r_amp):
            width = width_0  - df_de * gradE_dr
            spectrum += amp*self._lineshape_func(freq, fpos, width, **kwargs)
        spectrum *= scale_factor
        return spectrum
    
    def bg_drifts(self, 
                  freq: NDArray, 
                  params: Parameters,
                  poly_terms: int = 2,
                  **kwargs):
        coefs = [params[f'b{i}'].value for i in range(poly_terms) if params[f'b{i}'] is not None]
        poly = np.poly1d(coefs)
        if len(coefs) < poly_terms:
            return np.zeros_like(freq)
        return poly(freq)

    def signal_with_bg(self, 
                       freq: NDArray, 
                       params: Parameters,
                       poly_terms: int = 3,
                       **kwargs) -> NDArray[np.float64]:
        signal = self.signal(freq, params, **kwargs)
        bg = self.bg_drifts(freq, params, poly_terms=poly_terms)
        return signal + bg

    def signal_with_bg_shifted(self, 
                       freq: NDArray, 
                       params: Parameters,
                       poly_terms: int = 3,
                       **kwargs) -> NDArray[np.float64]:
        f_shifted = freq - params['f_shift'].value
        signal = self.signal(f_shifted, params, **kwargs)
        bg = self.bg_drifts(f_shifted, params, poly_terms=poly_terms)
        return signal + bg
        
class FitModel():
    '''
    Class for fitting experimental EIT spectra using the SignalSimulator.
    '''
    def __init__(self, 
                 fit_func: Callable
                 ):
        self._fit_func = fit_func
        
    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        signal = self._fit_func(freq, params)
        difference = data - signal
        if data_err is None:
            return difference
        return difference / data_err

class DataFitter():
    def __init__(self, 
                 data: SpectralData,
                 model: FitModel):
        self._data = data
        self._model = model
    
    def set_data(self, data: SpectralData):
        self._data = data

    def fit(self, params: Parameters):
        result = minimize(self._model.residuals, 
                          params, 
                          args=(self._data.axes.f, 
                                self._data.signal, 
                                self._data.signal_err))
        return result


from sympy.physics.wigner import wigner_3j, wigner_6j

class RydbergStarkEIT:
    """
    Calculates relative EIT intensities for CW two-photon transitions 
    in the Hyperfine Paschen-Back regime (strong DC electric field).
    Assumes perpendicularly polarized lasers (x-axis).
    """
    
    def __init__(self, I=2.5, J_g=0.5, F_g=3, J_i=1.5, J_r=2.5):
        """
        Initialize the atomic system angular momenta.
        Default values are for 85Rb: 5S_1/2(F=3) -> 5P_3/2 -> nD_5/2
        """
        self.I = I
        self.J_g = J_g
        self.F_g = F_g
        self.J_i = J_i
        self.J_r = J_r

    def _hf_dipole(self, J1, F1, mF1, J2, F2, mF2, q):
        """Coupled to Coupled transition (Ground -> Intermediate)"""
        six_j = float(wigner_6j(J2, J1, 1, F1, F2, self.I))
        red_F = ((-1)**(J2 + self.I + F1 + 1) * np.sqrt((2*F2 + 1) * (2*F1 + 1)) * six_j)
        three_j = float(wigner_3j(F2, 1, F1, -mF2, q, mF1))
        
        return ((-1)**(F2 - mF2) * three_j * red_F)

    def _pb_dipole(self, Ji, Fi, mFi, Jr, mJr, mI, q):
        """Coupled to Uncoupled transition (Intermediate -> Rydberg)"""
        mJi = mFi - mI
        if abs(mJi) > Ji: 
            return 0.0
            
        cg_coeff = float((-1)**(Ji - self.I + mFi) * np.sqrt(2*Fi + 1) * wigner_3j(Ji, self.I, Fi, mJi, mI, -mFi))
        
        j_dipole = float((-1)**(Jr - mJr) * wigner_3j(Jr, 1, Ji, -mJr, q, mJi))
        
        return cg_coeff * j_dipole

    def _probe_drive_x(self, F_i, mFi, mFg):
        """Amplitude for x-polarized 780 nm probe"""
        d_minus = self._hf_dipole(self.J_g, self.F_g, mFg, self.J_i, F_i, mFi, -1)
        d_plus  = self._hf_dipole(self.J_g, self.F_g, mFg, self.J_i, F_i, mFi, 1)
        return (d_minus - d_plus) / np.sqrt(2)

    def _coupling_drive_x(self, F_i, mFi, mJr, mI):
        """Amplitude for x-polarized 480 nm coupling"""
        d_minus = self._pb_dipole(self.J_i, F_i, mFi, self.J_r, mJr, mI, -1)
        d_plus  = self._pb_dipole(self.J_i, F_i, mFi, self.J_r, mJr, mI, 1)
        return (d_minus - d_plus) / np.sqrt(2)

    def calculate_spectrum(self, detunings, gamma=6.0, pop_weights=None):
        """
        Calculates the relative intensities of the |m_J| Stark components.
        
        Args:
            detunings (dict): {F_i: detuning_in_MHz} for the intermediate states.
            gamma (float): Natural linewidth of the intermediate state (MHz).
            pop_weights (dict): {mFg: population_weight} to simulate optical pumping.
            
        Returns:
            dict: {|m_J|: relative_intensity} normalized to the maximum peak.
        """
        if pop_weights is None:
            # Default to an unpolarized thermal distribution
            pop_weights = {mFg: 1.0 for mFg in np.arange(-self.F_g, self.F_g + 1)}
            
        intensities_mJr = {}

        # Loop over initial ground states
        for mFg in np.arange(-self.F_g, self.F_g + 1):
            
            # Loop over uncoupled final Rydberg states
            for mJr in np.arange(-self.J_r, self.J_r + 1):
                for mI in np.arange(-self.I, self.I + 1):
                    
                    M_2photon = 0.0 + 0.0j
                    
                    # Coherent sum over intermediate hyperfine paths
                    for F_i, detuning in detunings.items():
                        for mFi in np.arange(-F_i, F_i + 1):
                            
                            d1 = self._probe_drive_x(F_i, mFi, mFg)
                            d2 = self._coupling_drive_x(F_i, mFi, mJr, mI)
                            
                            if d1 != 0 and d2 != 0:
                                complex_detuning = detuning - 1j * (gamma / 2.0)
                                M_2photon += (d1 * d2) / complex_detuning
                    
                    line_strength = np.abs(M_2photon)**2 * pop_weights.get(mFg, 0)
                    
                    if line_strength > 1e-8:
                        abs_mJr = round(abs(mJr), 1)
                        if abs_mJr not in intensities_mJr:
                            intensities_mJr[abs_mJr] = 0.0
                        intensities_mJr[abs_mJr] += line_strength

        # Normalize outputs
        max_val = max(intensities_mJr.values())
        return {mJ: val / max_val for mJ, val in sorted(intensities_mJr.items())}




#%%
if __name__=='__main__':
    # ----------------- Usage example ----------------------
    from pinqued_tools.spectroscopy.lineshapes import holtsmarkian
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    # Read reference Rydberg splittings
    ref_path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Scripts\\calculated_stark_maps\\stark_map_25D_MHz.csv'
    ref = FieldReference(ref_path)

    # define parameters of the spectrum
    params = {'efield': 0.6,
              'amp': 150, 
              'width_0': 30, 
              'gradE_dr': 2,
              'rel_amp': [0.6, 0.6, 1.0, 1.0, 1.0],
              'b0': 1e-3, 'b1': 1e-2, 'b3': 1e-4}
    
    params_sim = Parameters()
    for key, value in params.items():
        if key == 'rel_amp':
            continue
        params_sim.add(key, value=value)
    
    params_lmfit = Parameters()
    params_lmfit.add('efield', value=params['efield'], min=-0.1)
    params_lmfit.add('amp', value=params['amp']-20.0)
    params_lmfit.add('width_0', value=params['width_0']+10.0)
    params_lmfit.add('gradE_dr', value=params['gradE_dr']-1.0)


    # Instantiate signal simulator object
    sim = SignalSimulator(ref, holtsmarkian)


    # Generate detunings
    freq = np.linspace(200, -1500, 700)

    # Simulate signal
    signal = sim.signal(freq, params=params_sim, normalized=True)

    sigma = 1.0
    noise = np.random.normal(loc=0, scale=sigma, size=signal.shape)/(signal+1)
    signal_err =  sigma/(signal+1)
    signal_noise = signal + noise

    spectrum = SpectralData(signal=signal_noise, 
                            axes = Axes0D(f=freq),
                            signal_err=signal_err)

    fm = FitModel(sim)
    df = DataFitter(spectrum, fm).fit(params_lmfit)
    print(fit_report(df))
    print(df.params)

    # Plot results
    fig, ax = plt.subplots(figsize=(4,2))
    ax.set_title(f'Simulated EIT spectrum ($E = ${params["efield"]:.1f} V/cm)')
    ax.plot(freq, signal, linewidth=1.5)
    ax.fill_between(y1=signal, x=freq, y2=-2, color='C0', alpha=0.2)
    ax.scatter(x=freq, y=signal_noise, 
               marker='.', s=5,
               color='C3', alpha=0.5)
    # ax.plot(freq, df.best_fit, linewidth=1.5, color='C1')
    ef =  ref.interp(params['efield'])
    for label, amp, (fpos, _) in zip(ref.level_labels, params['rel_amp'], ef):
        ax.axvline(x=fpos, color='C3', linestyle='--')
    ax.set_xlabel('Detuning (MHz)')
    ax.set_ylabel('EIT Signal $S$ (%)')

# %%
