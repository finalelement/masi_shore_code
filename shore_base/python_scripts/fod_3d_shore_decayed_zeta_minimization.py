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
from dipy.reconst.shore import shore_matrix
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import pinv
from zeta_minimization import return_shore_coeffs_preds, return_shore_basis

# Define Paths for the data and bvals and bvecs
data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\multi_shell_data_b0.mat'
bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval'
bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec'

data_path = os.path.normpath(data_path)
bval_path = os.path.normpath(bval_path)
bvec_path = os.path.normpath(bvec_path)

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

print('Gradient Table Loaded and Ready for use')

data = loadmat(data_path)
actual_data = data['ms_data']

print('Input Data all loaded and ready for use ...')

# Define Paths for FOD Data
fod_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.mat'
fod_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bval'
fod_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bvec'

fod_data_path = os.path.normpath(fod_data_path)
fod_bval_path = os.path.normpath(fod_bval_path)
fod_bvec_path = os.path.normpath(fod_bvec_path)

fod_bvals, fod_bvecs = read_bvals_bvecs(fod_bval_path, fod_bvec_path)
fod_gtab = gradient_table(fod_bvals, fod_bvecs)

print('FOD Gradient Table Loaded and Ready for use')

fod_data = loadmat(fod_data_path)
fod_actual_data = fod_data['new_fod_q_space']

print('Output Data all loaded and ready for use ...')


# SHORE Regularization Matrix Initialization

def l_shore(radial_order):
    "Returns the angular regularisation matrix for SHORE basis"
    F = radial_order / 2
    n_c = int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, radial_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                diagL[counter] = (l * (l + 1)) ** 2
                counter += 1

    return np.diag(diagL)


def n_shore(radial_order):
    "Returns the angular regularisation matrix for SHORE basis"
    F = radial_order / 2
    n_c = int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))
    diagN = np.zeros(n_c)
    counter = 0
    for l in range(0, radial_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                diagN[counter] = (n * (n + 1)) ** 2
                counter += 1

    return np.diag(diagN)


print('Minimizing the zeta scale Parameter for Input Data ...')

def eval_shore(D, n, scale):
    lambdaN = 1e-8
    lambdaL = 1e-8
    radial_order = n

    Lshore = l_shore(radial_order)
    Nshore = n_shore(radial_order)

    M = shore_matrix(n, scale, gtab, 1./(4*np.pi**2))
    MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
    shorecoefs = np.dot(D, MpseudoInv.T)
    
    #shorecoefs = np.dot(D, pinv(Mshore).T)
    shorepred  = np.dot(shorecoefs, M.T)
    return np.linalg.norm ( D - shorepred )**2

zeta = minimize(lambda x: eval_shore(actual_data[:100,:], 6, x), 700.0)['x']
print(zeta)


print('Fitting Shore to both models')

radial_order = 6
# Zeta is estimated to be 1442 from non-linear minimization for entire Input Data set
zeta = 1442
lambdaN = 1e-8
lambdaL = 1e-8
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

savemat('shore_coeffs_r6_v2.mat',mdict={'input_shore': shore_input_matrix})

signal_recon = asmfit.fitted_signal()
mse_vector = np.zeros((57267,1))

for i in range(0,57267):

    orig_sig = actual_data[i,:]
    recon_sig = signal_recon[i,:]

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
plt.title('4-Shell Signal Representation Error using 3D-Shore')
maxfreq = n.max()
plt.show()


print('Shore Coefficients Fitted for Input Data !!')
# Zeta is estimated to be 1024 for entire FOD Data using non-linear minimization
radial_order = 6
zeta = 1024
lambdaN = 1e-8
lambdaL = 1e-8
#asm_fod = ShoreModel(fod_gtab, radial_order=radial_order,
#                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

#asm_fod_fit = asm_fod.fit(fod_actual_data)

shore_output_matrix, shore_output_preds = return_shore_coeffs_preds(fod_actual_data, radial_order, fod_gtab,zeta)

print('Shore Model Coeffs Generated for FOD ...')

print('Transforming object shape to save to a .mat file')

#shore_output_matrix = np.zeros((57267,50))

#for i in range(0,57267):

#    temp_shore_coeffs = asm_fod_fit.fit_array[i]._shore_coef
#    shore_output_matrix[i,:] = temp_shore_coeffs

#    if (i%1000 == 0):
#        print(i)

savemat('shore_coeffs_decayed_fod_r6.mat',mdict={'output_shore': shore_output_matrix})

print('Shore Coefficients Fitted for Output Data and Saved as well!!')

signal_recon = shore_output_preds
print('Signal Reconstructed from shore FOD coefficients')

mse_vector = np.zeros((57267,1))

for i in range(0,57267):

    orig_sig = fod_actual_data[i,:]
    recon_sig = signal_recon[i,:]

    temp_mse = np.mean((orig_sig - recon_sig)**2)
    mse_vector[i] = temp_mse

mse_vector[mse_vector >= 0.01] = 0.01

bin_size = 0.0001; min_edge = 0; max_edge = 0.01
N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)

n, bins, patches = plt.hist(x=mse_vector, bins=bin_list, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels - Total 57000')
plt.title('FOD Representation using 3D-Shore')
maxfreq = n.max()
plt.show()

print('Debug and save the Shore basis matrix for the FOD part')

#a = []
#store_cache = asm_fod_fit.model._cache
#for each in store_cache:
#    a.append(each)

#shore_fod_basis = store_cache[a[0]]
shore_fod_basis = return_shore_basis(radial_order,fod_gtab,zeta)
savemat('shore_fod_decayed_basis_r6.mat',mdict={'shore_basis': shore_fod_basis})