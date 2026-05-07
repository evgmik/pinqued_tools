import numpy as np

from pinqued_tools.spectroscopy.spectrum import SpectralData
from pinqued_tools.spectroscopy.field_reference import FieldReference
from pinqued_tools.spectroscopy.signal_simulator import GPPoissonSignalSimulator1D

from typing import Callable
from lmfit import minimize, Parameters
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy.interpolate import BSpline
from scipy.sparse import diags
from scipy.integrate import cumulative_trapezoid

import matplotlib.pyplot as plt






class GPPoissonModel1D():
    def __init__(self, 
                 data: SpectralData,
                 field_ref: FieldReference,
                 signal_sim: GPPoissonSignalSimulator1D,
                 E_vec_init: NDArray,
                 l=1.0, sigma_f=50.0
                 ):
        """
        z: spatial coordinates (mm)
        freq: frequency axis (MHz)
        spectra: 2D array [z_bins, freq_bins]
        l: correlation lengthscale (related to Debye length)
        sigma_f: prior variance for potential
        """
        self.field_ref = field_ref
        self.signal_sim = signal_sim
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
        # Integrate the initial macroscopic Electric Field guess to get the potential phi
        # NOTE: E_vec_init is in V/cm, x is in mm. We divide by 10 to get V/mm for integration.
        self.phi_vec_init = -cumulative_trapezoid(E_vec_init / 10.0, self.x, initial=0.0)
        self.f = data.axes.f
        spectra = data.signal
        self.data_max = np.max(spectra) # Use a single global max for normalization
        self.data = spectra / self.data_max
        self.px_size = np.abs(self.x[1] - self.x[0]) # Pixel size
        
        # 1. Bake in Poisson Logic: GP Prior on Potential phi(x)
        # We use a Matern 5/2 kernel because it is twice differentiable.
        self.K_phi = self._matern_kernel(self.x, self.x, l, sigma_f)
        # Regularize for inversion
        self.K_inv = np.linalg.inv(self.K_phi + 1e-6 * np.eye(len(self.x)))
        
        # Cholesky decomposition of K_inv to formulate the GP prior as sum of squares
        # (L_inv @ phi).T @ (L_inv @ phi) = phi.T @ K_inv @ phi
        self.L_inv = cholesky(self.K_inv, lower=False)

    def _matern_kernel(self, x1, x2, l, sigma_f):
        d = np.abs(x1[:, None] - x2[None, :])
        arg = np.sqrt(5) * d / l
        return sigma_f**2 * (1 + arg + (arg**2)/3) * np.exp(-arg)

    def setup_params(self, base_params: Parameters) -> Parameters:
        """
        Adds the phi_vec parameters to an existing lmfit Parameters object.
        """
        params = base_params.copy()
        num_phi = len(self.phi_vec_init)
        
        # Prevent vanishing gradient if initialized with exact zeros
        if np.allclose(self.phi_vec_init, 0.0):
            # Assuming bulk plasma (zero potential) is at x[0]
            self.phi_vec_init = -5.0 * np.abs(self.x - self.x[0])
            
        for i, phi_val in enumerate(self.phi_vec_init):
            params.add(f'phi_{i}', value=phi_val)
            
        # Enforce phi -> 0 and E -> 0 at the boundary deep in the plasma.
        if num_phi > 1:
            params[f'phi_{num_phi - 1}'].set(value=0, vary=False)
            params[f'phi_{num_phi - 2}'].set(value=0, vary=False)
        elif num_phi == 1:
            params[f'phi_{num_phi - 1}'].set(value=0, vary=False)
        
        # Ensure efield and grad_vec exist in params since signal_sim expects them
        if 'efield' not in params:
            params.add('efield', value=0.0, vary=False)
        if 'grad_vec' not in params:
            params.add('grad_vec', value=0.0, vary=False)
            
        return params

    def forward_physics(self, params):
        """Maps Potential -> Field -> Smeared Spectra."""
        # Reconstruct phi_vec array from the lmfit parameters
        phi_vec = np.array([params[f'phi_{i}'].value for i in range(len(self.x))])
        
        # E = -d_phi/dx. Spatial axis `x` is in mm, so gradient is V/mm.
        # Multiply by 10.0 to convert to V/cm (required by Stark reference).
        E_vec = -np.gradient(phi_vec, self.x) * 10.0
        # grad = dE/dx (Broadening driver) in (V/cm) / mm
        grad_vec = np.gradient(E_vec, self.x)
        
        # Fully vectorized 2D Spectrum Evaluation
        S_pred = self.signal_sim.holtsmark_spectrum(self.f, params, 
                                                    efield=E_vec, grad_vec=grad_vec)
        
        # Ensure predicted spectrum orientation matches experimental data
        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec

    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        S_pred, E_vec, _, phi_vec = self.forward_physics(params)
        
        # Normalize data identically to internal logic
        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
        else:
            data_err_norm = data_err / self.data_max
            data_res = (difference / data_err_norm).flatten()
            
        # Include GP smoothness penalty as "prior residuals"
        prior_res = self.L_inv @ phi_vec

        # Soft penalty to bound E-field without zeroing LM gradients
        max_E = np.max(self.field_ref.efield)
        overshoot = np.clip(E_vec - max_E, 0, None)
        undershoot = np.clip(-E_vec, 0, None)
        # Smooth quadratic penalty to prevent infinite Jacobian walls that break the optimizer
        bounds_penalty = 1e4 * (overshoot**2 + undershoot**2)
        
        return np.concatenate([data_res, prior_res, bounds_penalty])


class BSplinePoissonModel1D():
    def __init__(self, 
                 data: SpectralData, #Input Stark map with frequency and 1 spatial axis
                 field_ref: FieldReference, # FieldReference instance to calculate Stark shifts
                 signal_sim: GPPoissonSignalSimulator1D, # Spectral signal simulator
                 E_vec_init: NDArray, # Initial guess of E-field distribution
                 E0_vec_init: NDArray|None = None, # Initial guess for Holtsmark field distribution
                 n_splines: int = 25, # Dimension of spline basis
                 spline_degree: int = 3,
                 smooth_param: float = 1e4, # Smoothing parameter for E-field spline
                 smooth_param_E0: float = 1e4 # Smoothing parameter for Holtsmark field spline
                 ):
        """
        Models the potential phi(x) using penalized B-splines (P-splines).

        z: spatial coordinates (mm)
        freq: frequency axis (MHz)
        spectra: 2D array [z_bins, freq_bins]
        n_splines: number of B-spline basis functions.
        spline_degree: degree of the B-spline (e.g., 3 for cubic).
        smooth_param: smoothing penalty weight for the potential phi. 
                      (Often needs to be 1e3 - 1e6 to overpower data noise).
        smooth_param_E0: smoothing penalty weight for the microfield E0.
                         (Often needs to be 1e3 - 1e6).
        """
        self.field_ref = field_ref
        self.signal_sim = signal_sim
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
        if len(E_vec_init) != len(self.x):
            raise ValueError(f"Length of E_vec_init ({len(E_vec_init)}) must match spatial axis length ({len(self.x)}).")
        # Integrate the initial macroscopic Electric Field guess to get the potential phi
        # Note: E_vec_init is in V/cm, x is in mm. We divide by 10 to get V/mm for integration.
        self.phi_vec_init = -cumulative_trapezoid(E_vec_init / 10.0, self.x, initial=0.0)
        self.f = data.axes.f
        spectra = data.signal
        
        if E0_vec_init is None:
            self.E0_vec_init = np.full_like(self.x, 3.0)
        else:
            if len(E0_vec_init) != len(self.x):
                raise ValueError(f"Length of E0_vec_init ({len(E0_vec_init)}) must match spatial axis length ({len(self.x)}).")
            self.E0_vec_init = E0_vec_init
            
        self.n_splines = n_splines
        self.smooth_param = smooth_param
        self.smooth_param_E0 = smooth_param_E0
        self.data_max = np.max(spectra) # Use a single global max for normalization
        self.data = spectra / self.data_max
        self.px_size = np.abs(self.x[1] - self.x[0]) # Pixel size
        
        # 1. Bake in P-Spline Logic: Penalized B-Spline Prior on Potential phi(x)
        self.k = spline_degree
        if n_splines <= self.k:
            raise ValueError("Number of splines must be greater than spline degree.")
        
        # Define knots for the B-spline basis. Use clamped knots for well-behaved boundaries.
        # For n_splines basis functions of degree k, we need n_splines - k - 1 interior knots.
        n_internal_knots = n_splines - self.k - 1
        internal_knots = np.linspace(self.x[0], self.x[-1], n_internal_knots + 2)[1:-1]
        self.knots = np.concatenate(([self.x[0]] * (self.k + 1), internal_knots, [self.x[-1]] * (self.k + 1)))

        # Construct the B-spline basis matrix B, where phi = B @ c
        B = np.zeros((len(self.x), n_splines))
        for i in range(n_splines):
            c = np.zeros(n_splines)
            c[i] = 1
            # BSpline is defined by knots, coefficients, and degree.
            spl = BSpline(self.knots, c, self.k, extrapolate=False)
            B[:, i] = spl(self.x)
        
        # Get pseudo-inverse to map from potential phi to spline coefficients c
        self.B_plus = np.linalg.pinv(B)
        self.c_init = self.B_plus @ self.phi_vec_init
        self.c_E0_init = self.B_plus @ self.E0_vec_init
        
        # Construct difference matrix for penalty on coefficients.
        # A 3rd-order difference penalizes the 2nd derivative of the E-field,
        # allowing the optimizer to form physically realistic linear E-fields (sheaths)
        # with ZERO penalty, eliminating the artificial "curved up" parabola effect. 
        # NOTE: (DID NOT WORK)
        if self.n_splines >= 4:
            self.D = diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(self.n_splines - 3, self.n_splines)).toarray()
        elif self.n_splines >= 3:
            self.D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(self.n_splines - 2, self.n_splines)).toarray()
        else:
            self.D = diags([-1.0, 1.0], [0, 1], shape=(self.n_splines - 1, self.n_splines)).toarray()



    def setup_params(self, base_params: Parameters) -> Parameters:
        """
        Adds the spline coefficient parameters to an existing lmfit Parameters object.
        """
        params = base_params.copy()
        
        # Add spatially varying background parameters (independent for each spatial pixel)
        for i in range(len(self.x)):
            if f'b0_{i}' not in params:
                params.add(f'b0_{i}', value=1e-4) # Slope
            if f'b1_{i}' not in params:
                params.add(f'b1_{i}', value=1e-4) # Offset
            if f'amp_{i}' not in params:
                init_amp = params['amp'].value if 'amp' in params else 100.0
                params.add(f'amp_{i}', value=init_amp, min=0.0) # Amplitude
        # Model background and amplitude using B-splines to drastically reduce parameter count
        init_amp = params['amp'].value if 'amp' in params else 100.0
        for i in range(self.n_splines):
            if f'c_b0_{i}' not in params:
                params.add(f'c_b0_{i}', value=1e-4)
            if f'c_b1_{i}' not in params:
                params.add(f'c_b1_{i}', value=1e-4)
            if f'c_amp_{i}' not in params:
                params.add(f'c_amp_{i}', value=init_amp, min=0.0)

        c_init = self.c_init.copy()
        c_E0_init = self.c_E0_init.copy()
        # Prevent vanishing gradient if initialized with exact zeros
        if np.allclose(c_init, 0.0):
            # Assuming bulk plasma (zero potential) is at x[0]
            phi_slope = -5.0 * np.abs(self.x - self.x[0])
            c_init = self.B_plus @ phi_slope

        # Enforce monotonically decreasing potential phi(x) by constraining B-spline 
        # coefficients. This strictly guarantees Electric Field >= 0 everywhere.
        params.add(f'c_{self.n_splines - 1}', value=0.0, vary=False)
        for i in range(self.n_splines - 2, -1, -1):
            delta_init = max(0.0, c_init[i] - c_init[i+1])
            params.add(f'delta_c_{i}', value=delta_init, min=0.0)
            params.add(f'c_{i}', expr=f'c_{i+1} + delta_c_{i}')
            
        # Constrain E0 to a physical ceiling
        for i in range(self.n_splines):
            params.add(f'c_{i}', value=c_init[i])
            # Constrain E0 to a physical ceiling to prevent it from growing
            # arbitrarily large and mimicking macroscopic Stark splitting.
            params.add(f'c_E0_{i}', value=c_E0_init[i], min=1e-3, max=25.0, vary=False)
            
        # Enforce phi -> 0 and E -> 0 at the boundary deep in the plasma.
        # c_0 controls the boundary potential, and c_1 (relative to c_0) controls 
        # the boundary derivative. Locking both to 0 guarantees E = 0.
        # Enforce E -> 0 at the boundary deep in the plasma.
        if self.n_splines > 1:
            params[f'c_{self.n_splines - 1}'].set(value=0, vary=False)
            params[f'c_{self.n_splines - 2}'].set(value=0, vary=False)
        elif self.n_splines == 1:
            params[f'c_{self.n_splines - 1}'].set(value=0, vary=False)
            params[f'delta_c_{self.n_splines - 2}'].set(value=0.0, vary=False)
        
        if 'fshift' not in params:
            params.add('fshift', value=0.0)
        
        return params

    def forward_physics(self, params):
        """Maps Potential -> Field -> Smeared Spectra."""
        # Reconstruct B-spline coefficients c from the lmfit parameters
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
        
        # Construct continuous B-spline representation
        spl = BSpline(self.knots, c, self.k, extrapolate=False)
        spl_E0 = BSpline(self.knots, c_E0, self.k, extrapolate=False)
        
        # Evaluate potential and exact analytical derivatives from the B-spline
        phi_vec = spl(self.x)
        E0_vec = spl_E0(self.x)
        # E = -d_phi/dx (1st derivative, nu=1). 
        # Spatial axis `x` is in mm, so multiply by 10.0 to get V/cm.
        E_vec = -spl(self.x, nu=1) * 10.0
        # grad = dE/dx = -d2_phi/dx2 (2nd derivative, nu=2) in (V/cm) / mm
        grad_vec = -spl(self.x, nu=2) * 10.0
        
        # Extract spatially varying background coefficients
        b0_vec = np.array([params[f'b0_{i}'].value for i in range(len(self.x))])
        b1_vec = np.array([params[f'b1_{i}'].value for i in range(len(self.x))])
        amp_vec = np.array([params[f'amp_{i}'].value for i in range(len(self.x))])
        # Construct spatially varying background coefficients from splines
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        
        b0_vec = BSpline(self.knots, c_b0, self.k, extrapolate=False)(self.x)
        b1_vec = BSpline(self.knots, c_b1, self.k, extrapolate=False)(self.x)
        amp_vec = BSpline(self.knots, c_amp, self.k, extrapolate=False)(self.x)
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        # 2D Spectrum Evaluation
        S_pred = np.zeros((len(self.x), len(self.f)))
        for i in range(len(self.x)):
            S_pred[i,:] = self.signal_sim.holtsmark_spectrum_bg(f_shifted, params, 
                                                    efield=E_vec[i], grad_vec=grad_vec[i], E0=E0_vec[i],
                                                    b_coefs=[b0_vec[i], b1_vec[i]],
                                                    amp=amp_vec[i])

        # Ensure predicted spectrum orientation matches experimental data
        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params)

        if data.shape != S_pred.shape:
            raise ValueError(f"Data shape mismatch! The fitter provided data of shape {data.shape}, "
                             f"but the model evaluated a grid of shape {S_pred.shape}. "
                             "Ensure the DataFitter is initialized with the exact same SpectralData "
                             "object that was used to initialize the model.")

        # Normalize data identically to internal logic
        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
        else:
            data_err_norm = data_err / self.data_max
            data_res = (difference / data_err_norm).flatten()
            
        # Include P-spline smoothness penalty as "prior residuals"
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        
        prior_res = np.sqrt(self.smooth_param) * (self.D @ c)
        prior_res_E0 = np.sqrt(self.smooth_param_E0) * (self.D @ c_E0)
        # Small regularizing smoothing penalties for the background to keep it well-behaved
        prior_res_b0 = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_b0)
        prior_res_b1 = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_b1)
        prior_res_amp = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_amp)
        
        # Soft penalty to bound E-field without zeroing LM gradients
        # max_E = np.max(self.field_ref.efield)
        # overshoot = np.clip(E_vec - max_E, 0, None)
        # undershoot = np.clip(-E_vec, 0, None)
        # Smooth quadratic penalty to prevent infinite Jacobian walls that break the optimizer
        # bounds_penalty = 1e4 * (overshoot**2 + undershoot**2)
        
        return np.concatenate([data_res, prior_res, prior_res_E0])#, bounds_penalty])
        return np.concatenate([data_res, prior_res, prior_res_E0, prior_res_b0, prior_res_b1, prior_res_amp])#, bounds_penalty])
