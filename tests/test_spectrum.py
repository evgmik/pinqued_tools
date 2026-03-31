from conftest import *

def test_data_spectrum_initialization(sample_data_spectrum):
    """Ensure the cube stores data and axes correctly."""
    assert sample_data_spectrum.signal.shape == (200,)
    assert sample_data_spectrum.signal_err.shape == (200,)
    assert sample_data_spectrum.axes.units["f"] == "MHz"
    assert sample_data_spectrum.units["signal"] == "%"
    assert sample_data_spectrum.units["signal_err"] == "%"
    assert sample_data_spectrum.metadata["experiment"] == "test_001"

def test_data_strip_initialization(sample_data_strip):
    """Ensure the cube stores data and axes correctly."""
    assert sample_data_strip.signal.shape == (11, 200)
    assert sample_data_strip.signal_err.shape == (11, 200)
    assert sample_data_strip.axes.units["f"] == "MHz"
    assert sample_data_strip.units["signal"] == "%"
    assert sample_data_strip.units["signal_err"] == "%"
    assert sample_data_strip.metadata["experiment"] == "test_002"

def test_data_cube_initialization(sample_data_cube):
    """Ensure the cube stores data and axes correctly."""
    assert sample_data_cube.signal.shape == (11, 11, 200)
    assert sample_data_cube.signal_err.shape == (11, 11, 200)
    assert sample_data_cube.axes.units["x"] == "mm"
    assert sample_data_cube.axes.units["y"] == "mm"
    assert sample_data_cube.axes.units["f"] == "MHz"
    assert sample_data_cube.units["signal"] == "%"
    assert sample_data_cube.units["signal_err"] == "%"
    assert sample_data_cube.metadata["experiment"] == "test_003"
