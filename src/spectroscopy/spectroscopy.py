#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%

def gaussian(freq: np.ndarray, 
             fpos: float = 0.0, # Units of `freq`
             width: float = 20.0, # Units of `freq`
             amplitude: float = 1.0,
             normalized: bool = True) -> np.ndarray:
    '''
    Normalized Gaussian lineshape
    '''
    sigma = width / np.sqrt(8*np.log(2))
    norm = 1.0 / np.sqrt(2 * np.pi * sigma)
    shape = np.exp( - 0.5 * ((freq - fpos) / sigma)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def lorentzian(freq: np.ndarray, 
              fpos: float = 0.0, # Units of `freq`
              width: float = 20.0, # Units of `freq`
              amplitude: float = 1.0,
              normalized: bool = True) -> np.ndarray:
    '''
    Normalized Lorentzian lineshape
    '''
    norm = 2.0 * width / np.pi 
    shape = 1.0 / (1.0 + (2.0 * (freq - fpos) / width)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def holtsmarkian(freq: np.ndarray, 
                fpos: float = 0.0, # Units of `freq`
                width: float = 20.0, # Units of `freq`
                amplitude: float = 1.0,
                normalized: bool = True) -> np.ndarray:
    '''
    Normalized Holtsmark lineshape
    '''
    norm = (5.0/(2.0*np.pi))*np.sin(2.0*np.pi/5) / width
    arg = (2 * np.abs(freq - fpos) / width)**(2.5)
    shape = 1.0 / (1.0 + arg)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape

def lineshape(freq: np.ndarray, 
              params: dict):
    '''
    Any lineshape depending that is defined above
    the function itself must be passed as e.g. `func: lorentzian`
    '''
    shape_function = params['func']
    function_parameters = {k: v for k, v in params.items() if k != 'func'}
    return shape_function(freq, **function_parameters)

def simulate_spectrum(freq: np.ndarray, 
                      params: list[dict],
                      return_shapes: bool = False) -> dict|np.ndarray:
    '''
    Simulates a spectrum based on the set of lineshapes provided as functions
    withing the list of dictionaries `params`. If `return_shapes` is True, then 
    the function returns dict with the following entries:
     'shapes_list' containing separate spectral lines
     'spectrum' sum of the spectral lines i.e. total spectrum
    '''
    spectrum = np.zeros_like(freq)
    for p in params:
        spectrum += lineshape(x, p)
    if return_shapes:
        shapes = []
        for p in params:
            shape = lineshape(x,p)
            shapes.append(shape)
        return {'spectrum': spectrum, 'shapes_list': shapes}
    return spectrum

#%%
if __name__=='__main__':
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()
    params = [
        {'func': gaussian, 
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
    x = np.linspace(-100,100, 1000)

    labels = ['Gauss', 'Lorentz', 'Holtsmark']
    fig, ax = plt.subplots()
    for p, ll in zip(params, labels):
        y = lineshape(x, p)
        ax.plot(x,y, linewidth=1.5, label=ll)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('EIT Signal $S$ (arb. units)')
    ax.legend()
# %%
