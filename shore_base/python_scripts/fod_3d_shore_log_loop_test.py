from dipy.reconst.shore import ShoreModel
#from dipy.viz import window, actor
#from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from scipy.io import loadmat,savemat
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define Paths for the data and bvals and bvecs
data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ln_ms_data.mat'
bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals.bval'
bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs.bvec'

data_path = os.path.normpath(data_path)
bval_path = os.path.normpath(bval_path)
bvec_path = os.path.normpath(bvec_path)

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

print('Gradient Table Loaded and Ready for use')

data = loadmat(data_path)
actual_data = data['ln_dwmri_data']

print('Data all loaded and ready for use')

print('data.shape (%d, %d)' % actual_data.shape)

radial_order = 6
zeta = 3000
lambdaN = 1e-12
lambdaL = 1e-12
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

asmfit = asm.fit(actual_data)

print('Shore Model Coeffs Generated ...')

print('Transforming object shape to save to a .mat file')

shore_input_matrix = np.zeros((57267,50))

for i in range(0,57267):

    temp_shore_coeffs = asmfit.fit_array[i]._shore_coef
    shore_input_matrix[i,:] = temp_shore_coeffs

    if (i%1000 == 0):
        print(i)

signal_recon = asmfit.fitted_signal()

print('Signal Reconstructed from shore signal coefficients')

mse_vector = np.zeros((57267,1))

for i in range(0,57267):

    orig_sig = actual_data[i,:]
    orig_sig = np.exp(orig_sig)
    recon_sig = signal_recon[i,:]
    recon_sig = np.exp(recon_sig)
    temp_mse = np.mean((orig_sig - recon_sig)**2)
    mse_vector[i] = temp_mse

mse_vector[mse_vector>=0.01] = 0.01

bin_size = 0.0001; min_edge = 0; max_edge = 0.01
N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)

n, bins, patches = plt.hist(x=mse_vector, bins=bin_list, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels - Total 57000')
plt.title('Signal Representation using 3D-Shore')
maxfreq = n.max()
plt.show()

print('Second Round refitting Shore to reconstructed signal')

asm2 = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
asmfit_2 =asm2.fit(signal_recon)
signal_recon_2 = asmfit_2.fitted_signal()

print('Second round signal reconstructed')

mse_vector_2 = np.zeros((57267,1))

for i in range(0,57267):
    orig_sig = signal_recon[i, :]
    orig_sig = np.exp(orig_sig)
    recon_sig = signal_recon_2[i, :]
    recon_sig = np.exp(recon_sig)
    temp_mse = np.mean((orig_sig - recon_sig) ** 2)
    mse_vector_2[i] = temp_mse

n, bins, patches = plt.hist(x=mse_vector_2, bins=100, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels - Total 57000')
plt.title('Signal Representation using 3D-Shore Second Iteration')
maxfreq = n.max()
plt.show()



print('Third Round refitting Shore to reconstructed signal')

asm3 = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
asmfit_3 = asm3.fit(signal_recon_2)
signal_recon_3 = asmfit_3.fitted_signal()

print('Third round signal reconstructed')

mse_vector_3 = np.zeros((57267,1))

for i in range(0,57267):
    orig_sig = signal_recon_2[i, :]
    orig_sig = np.exp(orig_sig)
    recon_sig = signal_recon_3[i, :]
    recon_sig = np.exp(recon_sig)
    temp_mse = np.mean((orig_sig - recon_sig) ** 2)
    mse_vector_3[i] = temp_mse

n, bins, patches = plt.hist(x=mse_vector_3, bins=100, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels - Total 57000')
plt.title('Signal Representation using 3D-Shore Third Iteration')
maxfreq = n.max()
plt.show()

