# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from pinqued_tools.spectroscopy.spectroscopy import simulate_spectrum, holtsmarkian, gaussian
from pinqued_tools.spectroscopy.efield import reference_interp, read_reference, eit_signal


#%%

amp_rel = [1.35,
           1.35,
           np.sqrt(6),
           np.sqrt(6),
           np.sqrt(6)]
file_path = 'G:/My Drive/Vaults/WnM-AMO/__Scripts/calculated_stark_maps/stark_map_25D_MHz.csv'
ref = read_reference(file_path, amp_rel)



params = {'amp': 4e2,
          'width_0': 20.0,
          'gradE_dr': 10.0}
freq = np.linspace(100, -1.5e3, 638)
signal = eit_signal(freq, 
                    efield=25.0,
                    reference_dict=ref,
                    params_dict=params,
                    lineshape_func=holtsmarkian)
fig, ax = plt.subplots()
ax.plot(freq, signal)
ax.set_xlabel('Freq. detuning (MHz)')
ax.set_ylabel('EIT signal $S$ (arb. units)')
# %%
for field in np.arange(0,10,10):
    params = {'amp': 4e2,
              'width_0': 20.0,
              'gradE_dr': 10.0}
freq = np.linspace(100, -1.5e3, 638)
signal = eit_signal(freq, 
                    efield=25.0,
                    reference_dict=ref,
                    params_dict=params,
                    lineshape_func=holtsmarkian)
fig, ax = plt.subplots()
ax.plot(freq, signal)
ax.set_xlabel('Freq. detuning (MHz)')
ax.set_ylabel('EIT signal $S$ (arb. units)')




# %%

# Read and plot reconstruction

import h5py
from os.path import join


dir_path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-12-08\\reconstructions'
file_nums = [2, 22]

reconstr = []

for num in file_nums:
    file_name = f'reconstr2d-2025-10-21_{num}.h5'
    file_path = join(dir_path, file_name)
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())# List keys and access datasets
    
        field = f['/maps/E-field'][()] # Read dataset into NumPy array
        gradient = f['/maps/delta E'][()] # Read attributes
        x = f['/axes/X'][()]
        z = f['/axes/Z'][()]
        reconstr.append((field, gradient, x, z))
# %%
plt.imshow(reconstr[1][1])

charge_density = reconstr[1][1]- reconstr[0][1] 
plt.imshow(charge_density)
# %%
xshift = 5.26
fig, ax = plt.subplots(2,1, figsize=(5,6), sharex=True)
for i in np.arange(5,len(reconstr[0][3]),7):
    z_pos = reconstr[0][3][i]
    label = f'{z_pos:.2f}'
    x = reconstr[0][2]
    G = charge_density.T[:,i]
    ax[0].plot(x-xshift, reconstr[1][0].T[:,i], '.-')
    # ax[1].plot(x-xshift, 1e-3*reconstr[1][1].T[:,i]/30e-4, '.-', label=label)
    ax[1].plot(x-xshift, 1e-3*charge_density.T[:,i]/30e-4, '.-')

ax[0].set_title('E-field reconstruction')
ax[1].set_title('E-field gradient')
ax[1].legend(title='$z$ (mm)')
ax[1].set_xlabel('$x$ (mm)')
ax[0].set_ylabel('$|E|$ (V/cm)')
ax[1].set_ylabel('$|\\nabla E|$ ($\\times 10^3$ V/cm$^2$)')



# %%
