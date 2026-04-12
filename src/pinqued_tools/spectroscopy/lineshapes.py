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






class HoltsmarkLine(BaseSpectralLine):
    '''
    Holtsmark lineshape as a subclass of `BaseSpectralLine`
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
        
        # Pre-calculate the frequency shifts for the integration grid
        # This only happens once during initialization!
        self.shifts = self.stark_interp(self.E_grid)
        
        # 2. Pre-compute the Holtsmark H(beta) interpolator for speed
        betas = np.linspace(0, 20.0, 2000)
        h_vals = np.array([self._integrate_holtsmark(b) for b in betas])
        self.H_beta_interp = interp1d(betas, h_vals, kind='cubic', bounds_error=False, fill_value=0.0)

    def __call__(self, freq: NDArray, 
                 efield: float = 0.0, # Units of `freq`
                 width: float = 20.0, # Units of `freq`
                 E0: float = 3.0, # Units of electric field strength
                 amplitude: float = 1.0,
                 model: str = '2d') -> NDArray:
        if model == '1d':
            return self.line1d(freq, efield, width, E0, amplitude)
        elif model == '2d':
            return self.line2d(freq, efield, width, E0, amplitude)
        else:
            raise ValueError("Invalid model. Choose '1d' or '2d'.")

    def line1d(self, freq: NDArray, 
                 efield: float = 0.0, # Units of `freq`
                 width: float = 20.0, # Units of `freq`
                 E0: float = 3.0, # Units of electric field strength
                 amplitude: float = 1.0) -> NDArray:
        # 1. Calculate the Holtsmark probability weights for the E_grid
        # H(E) = (1/E_0) * H(beta) where beta = E / E_0
        betas = (self.E_grid - efield) / E0
        weights = (1.0 / E0) * self.H_beta_interp(betas) * self.dE
        
        # 2. Build the Lorentzian Matrix (Broadcasting magic happens here)
        # nu_grid becomes a column vector (N x 1)
        # self.shifts is a row vector (1 x M)
        # detunings matrix becomes shape (N x M)
        shifts = self.stark_interp(self.E_grid)
        nu_col = freq[:, np.newaxis]
        detunings = nu_col - self.shifts[np.newaxis, :]
        
        gamma_half = width / 2.0
        lorentzian_matrix = (gamma_half**2) / (detunings**2 + gamma_half**2)
        
        # 3. Perform the integration as a dot product
        # Multiply the (N x M) matrix by the (M,) weights vector to get an (N,) spectrum
        spectrum = np.dot(lorentzian_matrix, weights)
        
        # Normalize area
        spectrum /= np.trapz(spectrum, freq)
        
        return amplitude * spectrum
    
    def line2d(self, freq: NDArray, 
               efield: float = 0.0,  # External DC field magnitude
               width: float = 20.0,  # Homogeneous linewidth (gamma)
               E0: float = 3.0,      # Normal microfield (plasma density)
               amplitude: float = 1.0) -> NDArray:
        """
        Generates the lineshape using a 2D vector summation of the external 
        DC field and the isotropic Holtsmark microfield.
        """
        # Ensure theta grid exists (fallback if you haven't added it to __init__)
        if not hasattr(self, 'theta_grid'):
            self.theta_points = 20
            self.theta_grid = np.linspace(0, np.pi, self.theta_points)
            self.dTheta = self.theta_grid[1] - self.theta_grid[0]

        # 1. Evaluate Holtsmark ONLY on the microfield grid
        # self.E_grid represents Em (microfield magnitude) here
        betas = self.E_grid / E0
        H_vals = (1.0 / E0) * self.H_beta_interp(betas)

        # 2. Vector Math: Create 2D meshgrids for Em and Theta
        # Em_matrix shape: (M, T). theta_matrix shape: (M, T)
        Em_matrix, theta_matrix = np.meshgrid(self.E_grid, self.theta_grid, indexing='ij')

        # 3. Calculate Total Electric Field Magnitude (Law of Cosines)
        E_tot = np.sqrt(efield**2 + Em_matrix**2 + 2 * efield * Em_matrix * np.cos(theta_matrix))

        # 4. Get Stark shifts for the combined field
        E_tot_clear = np.clip(E_tot, self._efield_reference[0], self._efield_reference[-1])
        shifts_2d = self.stark_interp(E_tot_clear)

        # 5. Calculate joint probability weights 
        # Solid angle fraction for an isotropic field is: 0.5 * sin(theta) * dTheta
        weights_2d = H_vals[:, np.newaxis] * self.dE * 0.5 * np.sin(theta_matrix) * self.dTheta

        # Flatten the matrices to 1D vectors for lightning-fast broadcasting
        shifts_flat = shifts_2d.flatten()
        weights_flat = weights_2d.flatten()

        # 6. Build the Lorentzian Matrix (Broadcasting)
        nu_col = freq[:, np.newaxis]
        detunings = nu_col - shifts_flat[np.newaxis, :]

        gamma_half = width / 2.0
        lorentzian_matrix = (gamma_half**2) / (detunings**2 + gamma_half**2)

        # 7. Perform integration as a dot product
        spectrum = np.dot(lorentzian_matrix, weights_flat)

        # 8. Normalize area to 1, then apply global amplitude
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
    # Apply custom plotting style
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    # 1. Define list with parameters for each spectral line
    params = [
        {'func': gaussian, #<---- NOTE: spectral line function is a dict entry
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
        {'func': lorentzian, 
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
        {'func': holtsmarkian, 
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
    ]

    # 2. Frequency detunings -100 to 100 MHz
    x = np.linspace(-100,100,1000)

    # 3. Plot all available spectral lines
    labels = ['Gauss', 'Lorentz', 'Holtsmark']
    fig, ax = plt.subplots()
    for p, ll in zip(params, labels):
        y = lineshape(x, p) # Calculate spectral lineshape
        ax.plot(x,y, linewidth=1.5, label=ll)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('EIT Signal $S$ (arb. units)')
    ax.legend()
# %%