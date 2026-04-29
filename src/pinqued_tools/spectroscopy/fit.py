from pinqued_tools.spectroscopy.spectrum import SpectralData
from typing import Callable
from lmfit import minimize, Parameters
from numpy.typing import NDArray



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
                 model: FitModel|GPPoissonModel1D):
        self._data = data
        self._model = model
    
    def set_data(self, data: SpectralData):
        self._data = data

    def fit(self, params: Parameters, method: str = 'leastsq', **kwargs):
        result = minimize(self._model.residuals, 
                          params, 
                          method=method,
                          args=(self._data.axes.f, 
                                self._data.signal, 
                                self._data.signal_err),
                          **kwargs)
        return result