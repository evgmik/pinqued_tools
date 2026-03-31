import pytest
import numpy as np
from src.pinqued_tools.spectroscopy.spectrum import (
    Axes0D, Axes1D, Axes2D,
    SpectralDataSpec, SpectralDataStrip, SpectralDataCube
)

# Set a seed for reproducibility
np.random.seed(123)

@pytest.fixture
def sample_axes0d():
    """Provides a standard set of physical axes for testing."""
    return Axes0D(
        f=np.linspace(400, 800, 200), # 200 spectral points
        units={"f": "MHz"}
    )

@pytest.fixture
def sample_axes1d():
    """Provides a standard set of physical axes for testing."""
    return Axes1D(
        x=np.linspace(0, 10, 11),        # 11 pixels in x
        f=np.linspace(400, 800, 200), # 200 spectral points
        units={"x": "mm", "f": "MHz"}
    )

@pytest.fixture
def sample_axes2d():
    """Provides a standard set of physical axes for testing."""
    return Axes2D(
        x=np.linspace(0, 10, 11),        # 11 pixels in x
        y=np.linspace(0, 10, 11),        # 11 pixels in y
        f=np.linspace(400, 800, 200), # 200 spectral points
        units={"x": "mm", "y": "mm", "f": "MHz"}
    )

@pytest.fixture
def sample_data_spectrum(sample_axes0d):
    """Create a spectrum with a fake Gaussian peak in the center."""
    nf = len(sample_axes0d.f)
    signal = np.zeros(nf)
    
    peak = 10 * np.exp(-(sample_axes0d.f - 600)**2 / (2 * 20**2))
    signal = peak + np.random.poisson(lam=np.sqrt(peak))
    signal_err = np.sqrt(signal)

    return SpectralDataSpec(signal=signal, 
                    signal_err=signal_err, 
                    axes=sample_axes0d, 
                    units={"signal": "%", "signal_err": "%"},
                    metadata={"experiment": "test_001"})

@pytest.fixture
def sample_data_strip(sample_axes1d):
    """Create a 2D map with a fake Gaussian peak in the center."""
    nx, nf = len(sample_axes1d.x), len(sample_axes1d.f)
    signal = np.zeros((nx, nf))
    
    x_idx = 2
    peak = 10 * np.exp(-(sample_axes1d.f - 600)**2 / (2 * 20**2))
    signal[x_idx, :] = peak + np.random.poisson(lam=np.sqrt(peak))
    signal_err = np.sqrt(signal)

    return SpectralDataStrip(signal=signal, 
                         signal_err=signal_err, 
                         axes=sample_axes1d, 
                         units={"signal": "%", "signal_err": "%"},
                         metadata={"experiment": "test_002"})

@pytest.fixture
def sample_data_cube(sample_axes2d):
    """Create a 3D cube with a fake Gaussian peak in the center."""
    nx, ny, nf = len(sample_axes2d.x), len(sample_axes2d.y), len(sample_axes2d.f)
    signal = np.zeros((nx, ny, nf))
    
    # Add a peak at index [2, 2]
    x_idx, y_idx = 2, 2
    peak = 10 * np.exp(-(sample_axes2d.f - 600)**2 / (2 * 20**2))
    signal[x_idx, y_idx, :] = peak + np.random.poisson(lam=np.sqrt(peak))
    signal_err = np.sqrt(signal)

    return SpectralDataCube(signal=signal, 
                        signal_err=signal_err, 
                        axes=sample_axes2d, 
                        units={"signal": "%", "signal_err": "%"},
                        metadata={"experiment": "test_003"})

