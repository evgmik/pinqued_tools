from conftest import *

def test_spectrum_initialization(sample_spectrum):
    """Ensure the cube stores data and axes correctly."""
    assert sample_spectrum.signal.shape == (200,)
    assert sample_spectrum.signal_err.shape == (200,)
    assert sample_spectrum.axes.units["f"] == "MHz"
    assert sample_spectrum.units["signal"] == "%"
    assert sample_spectrum.units["signal_err"] == "%"
    assert sample_spectrum.metadata["experiment"] == "test_001"

def test_spectral_strip_initialization(sample_spectral_strip):
    """Ensure the cube stores data and axes correctly."""
    assert sample_spectral_strip.signal.shape == (11, 200)
    assert sample_spectral_strip.signal_err.shape == (11, 200)
    assert sample_spectral_strip.axes.units["f"] == "MHz"
    assert sample_spectral_strip.units["signal"] == "%"
    assert sample_spectral_strip.units["signal_err"] == "%"
    assert sample_spectral_strip.metadata["experiment"] == "test_002"

def test_spectral_cube_initialization(sample_cube):
    """Ensure the cube stores data and axes correctly."""
    assert sample_cube.signal.shape == (11, 11, 200)
    assert sample_cube.signal_err.shape == (11, 11, 200)
    assert sample_cube.axes.units["x"] == "mm"
    assert sample_cube.axes.units["y"] == "mm"
    assert sample_cube.axes.units["f"] == "MHz"
    assert sample_cube.units["signal"] == "%"
    assert sample_cube.units["signal_err"] == "%"
    assert sample_cube.metadata["experiment"] == "test_003"