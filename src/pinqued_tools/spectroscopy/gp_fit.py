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



class GPPoissonModel1D():
    def __init__(self, 
                 data: SpectralData,
                 field_ref: FieldReference,
                 signal_sim: GPPoissonSignalSimulator1D,
                 phi_vec_init: NDArray,
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
        self.phi_vec_init = phi_vec_init
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
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
        for i, phi_val in enumerate(self.phi_vec_init):
            params.add(f'phi_{i}', value=phi_val)
            
        # Enforce phi -> 0 and E -> 0 at the boundary deep in the plasma.
        # We assume this is the last point in the spatial array `x`.
        # This sets the potential reference and removes gauge invariance.
        # The E-field is the negative gradient of the potential. At the last point,
        # the gradient is computed via backward-difference, so E=0 implies that
        # the last two potential points are equal.
        if num_phi > 1:
            params[f'phi_{num_phi-1}'].set(value=0, vary=False)
            params[f'phi_{num_phi-2}'].set(value=0, vary=False)
        elif num_phi == 1:
            params[f'phi_0'].set(value=0, vary=False)
        
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
        
        # E = -d_phi/dx
        E_vec = -np.gradient(phi_vec, self.x)
        # grad = dE/dx (Broadening driver)
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
        S_pred, _, _, phi_vec = self.forward_physics(params)
        
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
        
        return np.concatenate([data_res, prior_res])


class BSplinePoissonModel1D():
    def __init__(self, 
                 data: SpectralData,
                 field_ref: FieldReference,
                 signal_sim: GPPoissonSignalSimulator1D,
                 phi_vec_init: NDArray,
                 n_splines: int = 25,
                 spline_degree: int = 3,
                 smooth_param: float = 1.0
                 ):
        """
        Models the potential phi(x) using penalized B-splines (P-splines).

        z: spatial coordinates (mm)
        freq: frequency axis (MHz)
        spectra: 2D array [z_bins, freq_bins]
        n_splines: number of B-spline basis functions.
        spline_degree: degree of the B-spline (e.g., 3 for cubic).
        smooth_param: smoothing penalty weight (lambda).
        """
        self.field_ref = field_ref
        self.signal_sim = signal_sim
        self.phi_vec_init = phi_vec_init
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
        self.f = data.axes.f
        spectra = data.signal
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
        
        # Construct second-order difference matrix for penalty on coefficients
        D = diags([1, -2, 1], [0, 1, 2], shape=(n_splines - 2, n_splines)).toarray()
        
        # Pre-calculate the penalty matrix P for the residuals.
        # The penalty term in the residuals will be sqrt(lambda) * D @ c,
        # where c = self.B_plus @ phi_vec.
        self.P = np.sqrt(smooth_param) * (D @ self.B_plus)

    def setup_params(self, base_params: Parameters) -> Parameters:
        """
        Adds the phi_vec parameters to an existing lmfit Parameters object.
        """
        params = base_params.copy()
        num_phi = len(self.phi_vec_init)
        
        phi_init = self.phi_vec_init.copy()
        # Prevent vanishing gradient if initialized with exact zeros
        if np.allclose(phi_init, 0.0):
            # Add a tiny linear slope so E = -grad(phi) is non-zero
            # Decays to 0 at the end to match boundary conditions
            phi_init = 1e-4 * np.abs(self.x - self.x[-1])

        for i, phi_val in enumerate(phi_init):
            params.add(f'phi_{i}', value=phi_val)
            
        # Enforce phi -> 0 and E -> 0 at the boundary deep in the plasma.
        # We assume this is the last point in the spatial array `x`.
        # This sets the potential reference and removes gauge invariance.
        # The E-field is the negative gradient of the potential. At the last point,
        # the gradient is computed via backward-difference, so E=0 implies that
        # the last two potential points are equal.
        if num_phi > 1:
            params[f'phi_{num_phi-1}'].set(value=0, vary=False)
            params[f'phi_{num_phi-2}'].set(value=0, vary=False)
        elif num_phi == 1:
            params[f'phi_0'].set(value=0, vary=False)
        
        return params

    def forward_physics(self, params):
        """Maps Potential -> Field -> Smeared Spectra."""
        # Reconstruct phi_vec array from the lmfit parameters
        phi_vec = np.array([params[f'phi_{i}'].value for i in range(len(self.x))])
        
        # Map potential array back to B-spline coefficients
        c = self.B_plus @ phi_vec
        
        # Construct continuous B-spline representation
        spl = BSpline(self.knots, c, self.k, extrapolate=False)
        
        # Evaluate exact analytical derivatives from the B-spline
        # E = -d_phi/dx (1st derivative, nu=1)
        E_vec = -spl(self.x, nu=1)
        # grad = dE/dx = -d2_phi/dx2 (2nd derivative, nu=2)
        grad_vec = -spl(self.x, nu=2)
        
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
        S_pred, _, _, phi_vec = self.forward_physics(params)
        
        # Normalize data identically to internal logic
        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
        else:
            data_err_norm = data_err / self.data_max
            data_res = (difference / data_err_norm).flatten()
            
        # Include P-spline smoothness penalty as "prior residuals"
        prior_res = self.P @ phi_vec
        
        return np.concatenate([data_res, prior_res])
