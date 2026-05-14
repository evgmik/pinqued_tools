#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from abc import ABC

from scipy.interpolate import interp1d
from scipy.integrate import quad

def gaussian(freq: NDArray, 
             fpos: float = 0.0, # Units of `freq`
             width: float = 20.0, # Units of `freq`
             amplitude: float = 1.0,
             normalized: bool = True) -> NDArray:
    '''
    Gaussian lineshape
    '''
    sigma = width / np.sqrt(8*np.log(2))
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
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
    norm = 2.0 / (np.pi * width)
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
from scipy.interpolate import interp1d, RegularGridInterpolator, CubicSpline
from numba import njit, prange

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

# --Holtsmark lineshape --
@njit(parallel=True, fastmath=True)
def _fast_lorentzian_sum(freq: NDArray, 
                         shifts_flat: NDArray, 
                         weights_flat: NDArray, 
                         width: float, 
                         amplitude: float = 1.0) -> NDArray:
    '''
    Fast Lorentzian sum using Numba JIT compilation.
    '''
    spectrum = np.zeros(freq.shape[0], dtype=np.float64)
    gamma_half = width / 2.0
    gamma_half_sq = gamma_half**2
    
    for i in prange(freq.shape[0]):
        val = 0.0
        f = freq[i]
        for j in range(shifts_flat.shape[0]):
            detuning = f - shifts_flat[j]
            val += weights_flat[j] * gamma_half_sq / (detuning**2 + gamma_half_sq)
        spectrum[i] = val
        
    # Analytical normalization over all space prevents artificial 
    # amplitude explosion when a peak shifts outside the frequency window
    # We omit np.sum(weights_flat) here to ensure that if the microfield 
    # distribution is truncated (e.g. at high E0), the missing probability mass 
    # correctly reduces the peak amplitude rather than artificially inflating it.
    total_area = np.pi * gamma_half 
      
    if total_area > 0:
        return amplitude * spectrum / total_area
        
    return amplitude * spectrum

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

        self.stark_interp = CubicSpline(efield_reference, stark_reference, extrapolate=False)
        self._efield_reference = efield_reference
        
        # Safe linear extrapolation up to 10x the calibration limit.
        # This prevents the Holtsmark tail from being truncated.
        max_ref_E = efield_reference[-1]
        self._dense_efield = np.linspace(efield_reference[0], max_ref_E * 10.0, 200000)
        self._dense_stark = np.zeros_like(self._dense_efield)
        
        valid = self._dense_efield <= max_ref_E
        self._dense_stark[valid] = self.stark_interp(self._dense_efield[valid])
        
        last_stark = self.stark_interp(max_ref_E)
        last_slope = self.stark_interp(max_ref_E, nu=1)
        self._dense_stark[~valid] = last_stark + last_slope * (self._dense_efield[~valid] - max_ref_E)
        
        # 1. Pre-compute Holtsmark distribution and its analytical integral C(u)
        betas = np.linspace(0, 20.0, 2000)
        h_vals = np.array([self._integrate_holtsmark(b) for b in betas])
        self.H_beta_interp = interp1d(betas, h_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        
        self._dense_betas = np.linspace(0, 100.0, 200000)
        self._dense_h_vals = self.H_beta_interp(self._dense_betas)
        
        # Extend analytical tail for beta > 20.0 (H(beta) -> 1.496 * beta^-2.5)
        tail_mask = self._dense_betas > 20.0
        self._dense_h_vals[tail_mask] = 1.496 / (self._dense_betas[tail_mask]**2.5)

        from scipy.integrate import cumulative_trapezoid
        integrand = np.zeros_like(self._dense_betas)
        integrand[1:] = self._dense_h_vals[1:] / self._dense_betas[1:]
        self._C_u_vals = cumulative_trapezoid(integrand, self._dense_betas, initial=0.0)

        self._lut_interpolator = None

    def _get_P_Etot(self, Etot_grid: NDArray, efield: float, E0: float) -> NDArray:
        """Exact analytical 1D projection of the 2D macroscopic + microfield sum."""
        E0 = max(E0, 1e-6)
        if efield < 1e-6:
            betas = Etot_grid / E0
            H_vals = np.interp(betas, self._dense_betas, self._dense_h_vals, left=0.0, right=0.0)
            return (1.0 / E0) * H_vals

        u_max = (Etot_grid + efield) / E0
        u_min = np.abs(Etot_grid - efield) / E0

        C_max = np.interp(u_max, self._dense_betas, self._C_u_vals, left=0.0, right=self._C_u_vals[-1])
        C_min = np.interp(u_min, self._dense_betas, self._C_u_vals, left=0.0, right=self._C_u_vals[-1])

        return (Etot_grid / (2.0 * efield * E0)) * (C_max - C_min)

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
        
        library = np.zeros((len(E0_grid), len(efield_grid), len(width_grid), len(freq_grid)))
        
        # Use an ultra-dense Etot grid to prevent any interpolation aliasing
        Etot_grid = np.linspace(0.0, self._dense_efield[-1], 20000)
        dEtot = Etot_grid[1] - Etot_grid[0]
        shifts_flat = np.interp(Etot_grid, self._dense_efield, self._dense_stark)

        if base_model == '2d':
            for j, efield in enumerate(efield_grid):
                for i, E0 in enumerate(E0_grid):
                    weights_flat = self._get_P_Etot(Etot_grid, efield, E0) * dEtot
                    
                    for k, width in enumerate(width_grid): 
                        library[i, j, k, :] = _fast_lorentzian_sum(freq_grid, shifts_flat, weights_flat, width, 1.0)
        else:
            for j, efield in enumerate(efield_grid):
                for i, E0 in enumerate(E0_grid):
                    E_m = Etot_grid - efield
                    valid_mask = E_m >= 0
                    betas = E_m / max(E0, 1e-6)
                    H_vals = np.interp(betas, self._dense_betas, self._dense_h_vals, left=0.0, right=0.0)
                    weights_flat = (1.0 / max(E0, 1e-6)) * H_vals * dEtot * valid_mask
                    
                    for k, width in enumerate(width_grid):
                        library[i, j, k, :] = _fast_lorentzian_sum(freq_grid, shifts_flat, weights_flat, width, 1.0)
                
        # 3. Create the 4-Dimensional Interpolator
        self._lut_interpolator = RegularGridInterpolator(
            (E0_grid, efield_grid, width_grid, freq_grid),
            library, 
            bounds_error=False, 
            fill_value=0.0
        )

        print("LUT Build Complete.")

    def save_lut(self, file_path: str) -> None:
        """
        Saves the generated 4D Look-Up Table to a compressed numpy .npz file.
        """
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` first.")
            
        grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
        np.savez_compressed(
            file_path,
            E0_grid=grid_E0,
            efield_grid=grid_E,
            width_grid=grid_W,
            freq_grid=grid_f,
            library=self._lut_interpolator.values
        )
        print(f"LUT successfully saved to {file_path}")

    def save_lut_hdf5(self, file_path: str, group_name: str = 'holtsmark_lut') -> None:
        """
        Saves the generated 4D Look-Up Table to an HDF5 file.
        """
        import h5py
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` first.")
            
        grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
        
        with h5py.File(file_path, 'a') as f:
            # Remove the group if it already exists to overwrite it cleanly
            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            group.create_dataset('E0_grid', data=grid_E0)
            group.create_dataset('efield_grid', data=grid_E)
            group.create_dataset('width_grid', data=grid_W)
            group.create_dataset('freq_grid', data=grid_f)
            group.create_dataset('library', data=self._lut_interpolator.values, compression='gzip', compression_opts=9)
        print(f"LUT successfully saved to {file_path} in group '{group_name}'")

    def line_lut(self, freq: NDArray, 
                 efield: float|NDArray = 0.0, 
                 width: float|NDArray = 20.0, 
                 E0: float|NDArray = 3.0, 
                 amplitude: float|NDArray = 1.0) -> NDArray:
        """
        Instant lineshape extraction from the pre-computed 4D Look-Up Table.
        Supports scalar inputs or 1D arrays for spatial variations.
        """
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` before using model='lut'.")
            
        efield = np.asarray(efield)
        width = np.asarray(width)
        E0 = np.asarray(E0)
        
        # Safely clip query parameters to the LUT grid boundaries to prevent out-of-bounds 
        # linear extrapolation, which can produce negative values or empty gaps in the spectra.
        grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
        E0 = np.clip(E0, grid_E0[0], grid_E0[-1])
        efield = np.clip(efield, grid_E[0], grid_E[-1])
        width = np.clip(width, grid_W[0], grid_W[-1])

        # Scalar parameters: return 1D frequency spectrum
        if efield.ndim == 0 and width.ndim == 0 and E0.ndim == 0:
            query_points = np.zeros((len(freq), 4))
            query_points[:, 0] = E0
            query_points[:, 1] = efield
            query_points[:, 2] = width
            query_points[:, 3] = freq 
            spectrum = self._lut_interpolator(query_points)
            return amplitude * spectrum
            
        # Array parameters (spatial grid): return 2D array (spatial x frequency)
        # Safely broadcast arrays before meshing
        E0_b, efield_b, width_b = np.broadcast_arrays(E0, efield, width)
        E0_grid, freq_grid = np.meshgrid(E0_b, freq, indexing='ij')
        efield_grid, _ = np.meshgrid(efield_b, freq, indexing='ij')
        width_grid, _ = np.meshgrid(width_b, freq, indexing='ij')
        
        query_points = np.zeros((efield_grid.size, 4))
        query_points[:, 0] = E0_grid.ravel()
        query_points[:, 1] = efield_grid.ravel()
        query_points[:, 2] = width_grid.ravel()
        query_points[:, 3] = freq_grid.ravel()
        
        spectrum = self._lut_interpolator(query_points)
        spectrum = spectrum.reshape(efield_grid.shape)
        
        if np.ndim(amplitude) > 0:
            amplitude = amplitude[:, np.newaxis]
            
        return amplitude * spectrum

    def __call__(self, freq: NDArray, 
                 efield: float|NDArray = 0.0, 
                 width: float|NDArray = 20.0, 
                 E0: float|NDArray = 3.0,     
                 amplitude: float|NDArray = 1.0,
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
        
        E0_safe = max(E0, 1e-6)
        E_max = min(efield + 20.0 * E0_safe, self._dense_efield[-1])
        Etot_grid = np.linspace(0.0, max(E_max, 1e-3), 10000)
        dEtot = Etot_grid[1] - Etot_grid[0]

        E_m = Etot_grid - efield
        valid_mask = E_m >= 0
        betas = E_m / E0_safe
        H_vals = np.interp(betas, self._dense_betas, self._dense_h_vals, left=0.0, right=0.0)
        weights_flat = (1.0 / E0_safe) * H_vals * dEtot * valid_mask

        shifts_flat = np.interp(Etot_grid, self._dense_efield, self._dense_stark)

        return _fast_lorentzian_sum(freq, shifts_flat, weights_flat, width, amplitude)
    
    def line2d(self, freq: NDArray, 
               efield: float = 0.0, 
               width: float = 20.0, 
               E0: float = 3.0, 
               amplitude: float = 1.0) -> NDArray:
        """
        Generates the lineshape using a 2D vector summation of the external 
        DC field and the isotropic Holtsmark microfield.
        """
        E0_safe = max(E0, 1e-6)
        E_max = min(efield + 20.0 * E0_safe, self._dense_efield[-1])
        Etot_grid = np.linspace(0.0, max(E_max, 1e-3), 10000)
        dEtot = Etot_grid[1] - Etot_grid[0]

        weights_flat = self._get_P_Etot(Etot_grid, efield, E0_safe) * dEtot
        shifts_flat = np.interp(Etot_grid, self._dense_efield, self._dense_stark)

        return _fast_lorentzian_sum(freq, shifts_flat, weights_flat, width, amplitude)
    
    def _integrate_holtsmark(self, beta):
        """Rigorous Holtsmark integral definition."""
        if beta == 0: return 0.0
        integrand = lambda x: x * np.sin(beta * x) * np.exp(-(x**1.5))
        result, _ = quad(integrand, 0, np.inf, limit=200)
        return (2.0 / np.pi) * result * beta
        

class HoltsmarkLineCupy(HoltsmarkLine):
    '''
    CuPy-accelerated Holtsmark lineshape.
    Stores the Look-Up Table on the GPU and evaluates it using 
    optimized `map_coordinates` to bypass SciPy's overhead.
    '''
    def build_lut(self, 
                  freq_grid: NDArray, 
                  efield_grid: NDArray, 
                  E0_grid: NDArray, 
                  width_grid: NDArray,
                  base_model: str = '2d'):
        # 1. Build CPU LUT first via the parent class
        super().build_lut(freq_grid, efield_grid, E0_grid, width_grid, base_model)
        
        # 2. Extract boundaries as native floats to prevent implicit GPU-syncs during np.clip
        self._grid_boundaries = [
            (float(E0_grid[0]), float(E0_grid[-1])),
            (float(efield_grid[0]), float(efield_grid[-1])),
            (float(width_grid[0]), float(width_grid[-1])),
            (float(freq_grid[0]), float(freq_grid[-1]))
        ]
        
        # 3. Precompute properties for fast physical-to-index mapping
        self._grid_mins_cp = cp.array([E0_grid[0], efield_grid[0], width_grid[0], freq_grid[0]])
        self._grid_maxs_cp = cp.array([E0_grid[-1], efield_grid[-1], width_grid[-1], freq_grid[-1]])
        self._grid_ns_cp = cp.array([len(E0_grid)-1, len(efield_grid)-1, len(width_grid)-1, len(freq_grid)-1])
        
        # 4. Transfer the library to the GPU
        print("Transferring LUT to GPU...")
        self._lut_cp = cp.asarray(self._lut_interpolator.values)
        print("GPU Transfer Complete.")

    def save_lut_hdf5(self, file_path: str, group_name: str = 'holtsmark_lut') -> None:
        """
        Saves the CuPy 4D Look-Up Table directly from the GPU to an HDF5 file.
        """
        import h5py
        if getattr(self, '_lut_cp', None) is None:
            raise RuntimeError("CuPy LUT not initialized. Call `build_lut()` first.")
            
        with h5py.File(file_path, 'a') as f:
            # Remove the group if it already exists to overwrite it cleanly
            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            
            # Fetch original grids from the base CPU interpolator if it exists
            if getattr(self, '_lut_interpolator', None) is not None:
                grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
            else:
                # Reconstruct grids natively from CuPy boundaries if CPU LUT was cleared
                bE0, bE, bW, bF = self._grid_boundaries
                ns = self._grid_ns_cp.get()
                grid_E0 = np.linspace(bE0[0], bE0[1], int(ns[0] + 1))
                grid_E  = np.linspace(bE[0], bE[1], int(ns[1] + 1))
                grid_W  = np.linspace(bW[0], bW[1], int(ns[2] + 1))
                grid_f  = np.linspace(bF[0], bF[1], int(ns[3] + 1))
                
            group.create_dataset('E0_grid', data=grid_E0)
            group.create_dataset('efield_grid', data=grid_E)
            group.create_dataset('width_grid', data=grid_W)
            group.create_dataset('freq_grid', data=grid_f)
            # Transfer array from GPU to CPU to be saved by h5py
            group.create_dataset('library', data=self._lut_cp.get(), compression='gzip', compression_opts=9)
        print(f"CuPy LUT successfully saved to {file_path} in group '{group_name}'")

    def line_lut(self, freq: NDArray, 
                 efield: float|NDArray = 0.0, 
                 width: float|NDArray = 20.0, 
                 E0: float|NDArray = 3.0, 
                 amplitude: float|NDArray = 1.0) -> NDArray:
        if getattr(self, '_lut_cp', None) is None:
            raise RuntimeError("CuPy LUT not initialized. Call `build_lut()` first.")
            
        bE0, bE, bW, bF = self._grid_boundaries
        E0 = np.clip(E0, bE0[0], bE0[1])
        efield = np.clip(efield, bE[0], bE[1])
        width = np.clip(width, bW[0], bW[1])

        # Move queries to GPU
        E0_cp = cp.asarray(E0)
        efield_cp = cp.asarray(efield)
        width_cp = cp.asarray(width)
        freq_cp = cp.asarray(freq)

        # Convert physical values to fractional indices natively on the GPU
        def to_idx(val, axis):
            vmin = self._grid_mins_cp[axis]
            vmax = self._grid_maxs_cp[axis]
            n = self._grid_ns_cp[axis]
            return (val - vmin) / (vmax - vmin) * n

        E0_idx = to_idx(E0_cp, 0)
        E_idx = to_idx(efield_cp, 1)
        W_idx = to_idx(width_cp, 2)
        f_idx = to_idx(freq_cp, 3)

        # Scalar evaluation
        if efield_cp.ndim == 0 and width_cp.ndim == 0 and E0_cp.ndim == 0:
            coords = cp.stack([
                cp.full_like(f_idx, E0_idx),
                cp.full_like(f_idx, E_idx),
                cp.full_like(f_idx, W_idx),
                f_idx
            ])
            spectrum_cp = map_coordinates(self._lut_cp, coords, order=1, mode='constant', cval=0.0)
            return np.asarray(amplitude) * spectrum_cp.get()
            
        # 2D Array evaluation
        E0_b, efield_b, width_b = cp.broadcast_arrays(E0_idx, E_idx, W_idx)
        E0_grid, freq_grid = cp.meshgrid(E0_b, f_idx, indexing='ij')
        efield_grid, _ = cp.meshgrid(efield_b, f_idx, indexing='ij')
        width_grid, _ = cp.meshgrid(width_b, f_idx, indexing='ij')
        
        coords = cp.stack([
            E0_grid.ravel(),
            efield_grid.ravel(),
            width_grid.ravel(),
            freq_grid.ravel()
        ])
        
        spectrum_cp = map_coordinates(self._lut_cp, coords, order=1, mode='nearest')
        spectrum = spectrum_cp.get().reshape(efield_grid.shape)
        
        if np.ndim(amplitude) > 0:
            amplitude = np.asarray(amplitude)[:, np.newaxis]
            
        return amplitude * spectrum

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