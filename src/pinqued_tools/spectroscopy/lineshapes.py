#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from abc import ABC

from scipy.interpolate import interp1d

def gaussian(freq: NDArray, 
             fpos: float = 0.0, # Units of `freq`
             width: float = 20.0, # Units of `freq`
             amplitude: float = 1.0,
             normalized: bool = True) -> NDArray:
    '''
    Gaussian lineshape
    '''
    sigma = width / np.sqrt(8*np.log(2))
    norm = 1.0 / np.sqrt(2 * np.pi * sigma)
    shape = np.exp( - 0.5 * ((freq - fpos) / sigma)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def lorentzian(freq: NDArray, 
              fpos: float = 0.0, # Units of `freq`
              width: float = 20.0, # Units of `freq`
              amplitude: float = 1.0,
              normalized: bool = True) -> NDArray:
    '''
    Lorentzian lineshape
    '''
    norm = 2.0 * width / np.pi 
    shape = 1.0 / (1.0 + (2.0 * (freq - fpos) / width)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def holtsmarkian(freq: NDArray, 
                fpos: float = 0.0, # Units of `freq`
                width: float = 20.0, # Units of `freq`
                amplitude: float = 1.0,
                normalized: bool = True) -> NDArray:
    '''
    Holtsmark lineshape
    '''
    norm = (5.0/(2.0*np.pi))*np.sin(2.0*np.pi/5) / width
    arg = (2 * np.abs(freq - fpos) / width)**(2.5)
    shape = 1.0 / (1.0 + arg)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape

def lineshape(freq: NDArray, 
              params: dict):
    '''
    Any lineshape depending that is defined above
    the function itself must be passed as e.g. `func: lorentzian`
    '''
    # Extract spectral lineshape function from dictioary
    shape_function = params['func']  
    # remove spectral lineshape function `params` dict 
    function_parameters = {k: v for k, v in params.items() if k != 'func'}
    # Use `shape_function` to generate lineshape using `function_parameters`
    return shape_function(freq, **function_parameters)

def simulate_spectrum(freq: NDArray, 
                      params: list[dict],
                      return_shapes: bool = False) -> dict|NDArray:
    '''
    Simulates a spectrum based on the set of lineshapes provided as functions
    withing the list of dictionaries `params`. If `return_shapes` is True, then 
    the function returns dict with the following entries:
     'shapes_list' containing separate spectral lines
     'spectrum' sum of the spectral lines i.e. total spectrum.
    Otherwise, only total spectrum is returned as an numpy array.
    '''
    spectrum = np.zeros_like(freq)
    for p in params:
        spectrum += lineshape(freq, p)
    if return_shapes:
        shapes = []
        for p in params:
            shape = lineshape(freq, p)
            shapes.append(shape)
        return {'spectrum': spectrum, 'shapes_list': shapes}
    return spectrum

# ------------------- COMPLICATED LINESHAPES -------------------

class BaseSpectralLine(ABC):
    '''
    Base class for spectral lineshapes. 
    '''
    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def __call__(self, 
                 freq: NDArray, 
                 fpos: float = 0.0, # Units of `freq`
                 width: float = 20.0, # Units of `freq`
                 amplitude: float = 1.0) -> NDArray:
        raise NotImplementedError("Subclasses must implement the __call__ method.")

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d, RegularGridInterpolator

class HoltsmarkLine(BaseSpectralLine):
    '''
    Holtsmark lineshape as a subclass of `BaseSpectralLine`
    Supports 1D scalar, 2D vector, and ultra-fast 3D Look-Up Table (LUT) models.
    '''
    def __init__(self, 
                 efield_reference: NDArray,
                 stark_reference: NDArray,
                 normalized: bool = True,
                 n_efield_points: int = 1000):
        super().__init__(normalized)

        self.stark_interp = interp1d(x=efield_reference, 
                                     y=stark_reference, 
                                     kind='cubic')
        self._efield_reference = efield_reference
        
        # 1. Define the integration grid for the Electric Field
        self.E_grid = np.linspace(1e-3, efield_reference[-1], n_efield_points)
        self.dE = self.E_grid[1] - self.E_grid[0]
        
        # 2. Pre-compute the Holtsmark H(beta) interpolator for speed
        betas = np.linspace(0, 20.0, 2000)
        h_vals = np.array([self._integrate_holtsmark(b) for b in betas])
        self.H_beta_interp = interp1d(betas, h_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        
        # 3. Placeholder for the LUT 3D Interpolator
        self._lut_interpolator = None

    def build_lut(self, 
                  freq_grid: NDArray, 
                  efield_grid: NDArray, 
                  E0_grid: NDArray, 
                  width_grid: NDArray,  # <-- Added width array
                  base_model: str = '2d'):
        """
        Pre-calculates a 4D Lineshape Library (E0, efield, width, freq) for instant fitting.
        """
        print(f"Building 4D Lineshape Library: {len(E0_grid)}x{len(efield_grid)}x{len(width_grid)}x{len(freq_grid)} points...")
        
        # 1. Allocate the 4D block of memory
        library = np.zeros((len(E0_grid), len(efield_grid), len(width_grid), len(freq_grid)))
        
        # 2. Populate the grid
        for i, E0 in enumerate(E0_grid):
            for j, efield in enumerate(efield_grid):
                for k, width in enumerate(width_grid):  # <-- Added width loop
                    
                    if base_model == '2d':
                        spectrum = self.line2d(freq=freq_grid, efield=efield, width=width, E0=E0, amplitude=1.0)
                    else:
                        spectrum = self.line1d(freq=freq_grid, efield=efield, width=width, E0=E0, amplitude=1.0)
                    
                    library[i, j, k, :] = spectrum
                
        # 3. Create the 4-Dimensional Interpolator
        self._lut_interpolator = RegularGridInterpolator(
            (E0_grid, efield_grid, width_grid, freq_grid),  # <-- 4-tuple
            library, 
            bounds_error=False, 
            fill_value=0.0
        )
        print("LUT Build Complete.")

    def line_lut(self, freq: NDArray, 
                 efield: float = 0.0, 
                 width: float = 20.0, 
                 E0: float = 3.0, 
                 amplitude: float = 1.0) -> NDArray:
        """
        Instant lineshape extraction from the pre-computed 4D Look-Up Table.
        """
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` before using model='lut'.")
            
        # Create a query array of shape (N, 4): [E0, efield, width, freq]
        query_points = np.zeros((len(freq), 4))
        query_points[:, 0] = E0
        query_points[:, 1] = efield
        query_points[:, 2] = width     # <-- Inject dynamic width here
        query_points[:, 3] = freq 
        
        # Extract the interpolated spectrum in microseconds
        spectrum = self._lut_interpolator(query_points)
        
        return amplitude * spectrum

    def __call__(self, freq: NDArray, 
                 efield: float = 0.0, # Units of electric field strength
                 width: float = 20.0, # Units of `freq`
                 E0: float = 3.0,     # Units of electric field strength
                 amplitude: float = 1.0,
                 model: str = 'lut') -> NDArray:
        
        if model == '1d':
            return self.line1d(freq, efield, width, E0, amplitude)
        elif model == '2d':
            return self.line2d(freq, efield, width, E0, amplitude)
        elif model == 'lut':
            return self.line_lut(freq, efield, width, E0, amplitude)
        else:
            raise ValueError("Invalid model. Choose '1d', '2d', or 'lut'.")

    def line1d(self, freq: NDArray, 
                 efield: float = 0.0, 
                 width: float = 20.0, 
                 E0: float = 3.0, 
                 amplitude: float = 1.0) -> NDArray:
        
        # [CORRECTED]: Evaluate Holtsmark ONLY on the pure microfield magnitude
        betas = self.E_grid / E0
        weights = (1.0 / E0) * self.H_beta_interp(betas) * self.dE
        
        # Calculate Total Electric Field (Scalar Approximation)
        E_tot = efield + self.E_grid
        
        # Clip to avoid interpolation errors and get Stark shifts
        E_tot_clear = np.clip(E_tot, self._efield_reference[0], self._efield_reference[-1])
        shifts = self.stark_interp(E_tot_clear)
        
        nu_col = freq[:, np.newaxis]
        detunings = nu_col - shifts[np.newaxis, :]
        
        gamma_half = width / 2.0
        lorentzian_matrix = (gamma_half**2) / (detunings**2 + gamma_half**2)
        
        spectrum = np.dot(lorentzian_matrix, weights)
        
        area = np.trapz(spectrum, freq)
        if area > 0:
            spectrum /= area
            
        return amplitude * spectrum
    
    def line2d(self, freq: NDArray, 
               efield: float = 0.0, 
               width: float = 20.0, 
               E0: float = 3.0, 
               amplitude: float = 1.0) -> NDArray:
        """
        Generates the lineshape using a 2D vector summation of the external 
        DC field and the isotropic Holtsmark microfield.
        """
        if not hasattr(self, 'theta_grid'):
            self.theta_points = 20
            self.theta_grid = np.linspace(0, np.pi, self.theta_points)
            self.dTheta = self.theta_grid[1] - self.theta_grid[0]

        betas = self.E_grid / E0
        H_vals = (1.0 / E0) * self.H_beta_interp(betas)

        Em_matrix, theta_matrix = np.meshgrid(self.E_grid, self.theta_grid, indexing='ij')

        E_tot = np.sqrt(efield**2 + Em_matrix**2 + 2 * efield * Em_matrix * np.cos(theta_matrix))

        E_tot_clear = np.clip(E_tot, self._efield_reference[0], self._efield_reference[-1])
        shifts_2d = self.stark_interp(E_tot_clear)

        weights_2d = H_vals[:, np.newaxis] * self.dE * 0.5 * np.sin(theta_matrix) * self.dTheta

        shifts_flat = shifts_2d.flatten()
        weights_flat = weights_2d.flatten()

        nu_col = freq[:, np.newaxis]
        detunings = nu_col - shifts_flat[np.newaxis, :]

        gamma_half = width / 2.0
        lorentzian_matrix = (gamma_half**2) / (detunings**2 + gamma_half**2)

        spectrum = np.dot(lorentzian_matrix, weights_flat)

        area = np.trapz(spectrum, freq)
        if area > 0:
            spectrum /= area

        return amplitude * spectrum
    
    def _integrate_holtsmark(self, beta):
        """Rigorous Holtsmark integral definition."""
        from scipy.integrate import quad
        if beta == 0: return 0.0
        integrand = lambda x: x * np.sin(beta * x) * np.exp(-(x**1.5))
        result, _ = quad(integrand, 0, np.inf, limit=200)
        return (2.0 / np.pi) * result
        
#%%
if __name__=='__main__':
# %%