from dipy.reconst.shore import ShoreModel
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
from zeta_minimization import returns_best_zeta, return_shore_coeffs_preds, return_mse_vector_coeffs_preds
from multiple_shore_fits_main_path_calls import call_minimization_shore_on_data_with_basis_mse
import matplotlib.pyplot as plt

# Global SHORE Hyper-Parameters
radial_order = 6
zeta_guess = 700.0
lambdaN = 1e-8
lambdaL = 1e-8

shore_para_dict = {'radial_order':6,'zeta_guess':700.0,'lambdaN':1e-8,'lambdaL':1e-8}

# Define Paths for the data and bvals and bvecs
# Single Shell fitting B3000, B6000, B9000 & B12000
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_data.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvals.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvecs.bvec'

b3k_shore_coeff, b3k_basis, b3k_mse = call_minimization_shore_on_data_with_basis_mse(b3k_data_path, b3k_bval_path, b3k_bvec_path, 'b3k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)
print(b3k_mse)

#b3k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_shore_coeffs.mat'
#savemat(b3k_shore_path,mdict={'shore_coeffs':b3k_shore_coeff})
print('I think the code worked, B3k is done !')

b6k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_data.mat'
b6k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_bvals.bval'
b6k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_bvecs.bvec'

b6k_shore_coeff, b6k_basis, b6k_mse = call_minimization_shore_on_data_with_basis_mse(b6k_data_path, b6k_bval_path, b6k_bvec_path, 'b6k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)
print(b6k_mse)

#b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_shore_coeffs.mat'
#savemat(b6k_shore_path,mdict={'shore_coeffs':b6k_shore_coeff})
print('B6k is done')

b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_data.mat'
b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_bvals.bval'
b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_bvecs.bvec'

b9k_shore_coeff, b9k_basis, b9k_mse = call_minimization_shore_on_data_with_basis_mse(b9k_data_path, b9k_bval_path, b9k_bvec_path, 'b9k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)
print(b9k_mse)

#b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_shore_coeffs.mat'
#savemat(b9k_shore_path,mdict={'shore_coeffs':b9k_shore_coeff})
print('B9k is done')

b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_data.mat'
b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvals.bval'
b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvecs.bvec'

b12k_shore_coeff, b12k_basis, b12k_mse = call_minimization_shore_on_data_with_basis_mse(b12k_data_path, b12k_bval_path, b12k_bvec_path, 'b12k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)
print(b12k_mse)

#b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b12k_shore_coeffs.mat'
#savemat(b12k_shore_path,mdict={'shore_coeffs':b12k_shore_coeff})
print('B12k is done')


# Two Shell Fitting (B3000,B6000) , (B3000, B9000) , (B3000, B12000) , (B6000,B9000) , (B6000,B12000) , (B9000,B12000)
b3k_b6k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_data.mat'
b3k_b6k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvals.bval'
b3k_b6k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvecs.bvec'

b3k_b6k_shore_coeff, b3k_b6k_basis, b3k_b6k_mse = call_minimization_shore_on_data_with_basis_mse(b3k_b6k_data_path, b3k_b6k_bval_path, b3k_b6k_bvec_path, 'b3k_b6k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

print(b3k_b6k_mse)
#b3k_b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_shore_coeffs.mat'
#savemat(b3k_b6k_shore_path,mdict={'shore_coeffs':b3k_b6k_shore_coeff})
print('B3k & B6k is done')


b3k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_data.mat'
b3k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_bvals.bval'
b3k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_bvecs.bvec'

b3k_b9k_shore_coeff, b3k_b9k_basis, b3k_b9k_mse = call_minimization_shore_on_data_with_basis_mse(b3k_b9k_data_path, b3k_b9k_bval_path, b3k_b9k_bvec_path, 'b3k_b9k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

print(b3k_b9k_mse)
#b3k_b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b9k_shore_coeffs.mat'
#savemat(b3k_b9k_shore_path,mdict={'shore_coeffs':b3k_b9k_shore_coeff})
print('B3k & B9k is done')


b3k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_data.mat'
b3k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_bvals.bval'
b3k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_bvecs.bvec'

b3k_b12k_shore_coeff, b3k_b12k_basis, b3k_b12k_mse = call_minimization_shore_on_data_with_basis_mse(b3k_b12k_data_path, b3k_b12k_bval_path, b3k_b12k_bvec_path, 'b3k_b12k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

print(b3k_b12k_mse)
#b3k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b12k_shore_coeffs.mat'
#savemat(b3k_b12k_shore_path,mdict={'shore_coeffs':b3k_b12k_shore_coeff})
print('B3k & B12k is done')


b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_data.mat'
b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_bvals.bval'
b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_bvecs.bvec'

b6k_b9k_shore_coeff, b6k_b9k_basis, b6k_b9k_mse = call_minimization_shore_on_data_with_basis_mse(b6k_b9k_data_path, b6k_b9k_bval_path, b6k_b9k_bvec_path, 'b6k_b9k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

print(b6k_b9k_mse)
#b6k_b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_b9k_shore_coeffs.mat'
#savemat(b6k_b9k_shore_path,mdict={'shore_coeffs':b6k_b9k_shore_coeff})
print('B6k & B9k is done')


b6k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_data.mat'
b6k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_bvals.bval'
b6k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_bvecs.bvec'

b6k_b12k_shore_coeff, b6k_b12k_basis, b6k_b12k_mse = call_minimization_shore_on_data_with_basis_mse(b6k_b12k_data_path, b6k_b12k_bval_path, b6k_b12k_bvec_path, 'b6k_b12k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

print(b6k_b12k_mse)
#b6k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_b12k_shore_coeffs.mat'
#savemat(b6k_b12k_shore_path,mdict={'shore_coeffs':b6k_b12k_shore_coeff})
print('B6k & B12k is done')


b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_data.mat'
b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_bvals.bval'
b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_bvecs.bvec'

b9k_b12k_shore_coeff, b9k_b12k_basis, b9k_b12k_mse = call_minimization_shore_on_data_with_basis_mse(b9k_b12k_data_path, b9k_b12k_bval_path, b9k_b12k_bvec_path, 'b9k_b12k_data', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

#b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_b12k_shore_coeffs.mat'
#savemat(b9k_b12k_shore_path,mdict={'shore_coeffs':b9k_b12k_shore_coeff})
print('B9k & B12k is done')


# Three Shell Fitting (b3000, b6000, b9000) , (b3000, b9000, b12000) , (b6000, b9000, b12000)


b3k_b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_data.mat'
b3k_b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvals.bval'
b3k_b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvecs.bvec'

b3k_b6k_b9k_shore_coeff, b3k_b6k_b9k_basis, b3k_b6k_b9k_mse = call_minimization_shore_on_data_with_basis_mse(b3k_b6k_b9k_data_path, b3k_b6k_b9k_bval_path, b3k_b6k_b9k_bvec_path, 'b3k_b6k_b9k_data', 1, shore_para_dict)

b3k_b6k_b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_b9k_shore_coeffs.mat'
savemat(b3k_b6k_b9k_shore_path,mdict={'shore_coeffs':b3k_b6k_b9k_shore_coeff})
print('B3k & B6k & B9K is done')


b3k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_data.mat'
b3k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_bvals.bval'
b3k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_bvecs.bvec'

b3k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b3k_b9k_b12k_data_path, b3k_b9k_b12k_bval_path, b3k_b9k_b12k_bvec_path, 'b3k_b9k_b12k_data', 1, shore_para_dict)

b3k_b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b9k_b12k_shore_coeffs.mat'
savemat(b3k_b9k_b12k_shore_path,mdict={'shore_coeffs':b3k_b9k_b12k_shore_coeff})
print('B3k & B9k & B12K is done')

b6k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_data.mat'
b6k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_bvals.bval'
b6k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_bvecs.bvec'

b6k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b6k_b9k_b12k_data_path, b6k_b9k_b12k_bval_path, b6k_b9k_b12k_bvec_path, 'b6k_b9k_b12k_data', 1, shore_para_dict)

b6k_b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_b9k_b12k_shore_coeffs.mat'
savemat(b6k_b9k_b12k_shore_path,mdict={'shore_coeffs':b6k_b9k_b12k_shore_coeff})
print('B6k & B9k & B12K is done')


b3k_b6k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_data.mat'
b3k_b6k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvals.bval'
b3k_b6k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvecs.bvec'

b3k_b6k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b3k_b6k_b9k_b12k_data_path, b3k_b6k_b9k_b12k_bval_path, b3k_b6k_b9k_b12k_bvec_path, 'b3k_b6k_b9k_b12k_data', 1, shore_para_dict)

b3k_b6k_b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_b9k_b12k_shore_coeffs.mat'
savemat(b3k_b6k_b9k_b12k_shore_path,mdict={'shore_coeffs':b3k_b6k_b9k_b12k_shore_coeff})
print('B3k & B6k & B9k & B12K is done')

















mse_save_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\mse_zeta_optimized.mat'
savemat(mse_save_path,mdict={'b3k_mse':b3k_mse,'b6k_mse':b6k_mse,'b9k_mse':b9k_mse,'b12k_mse':b12k_mse,'b3k_b6k_mse':b3k_b6k_mse,'b3k_b9k_mse':b3k_b9k_mse,})

mse_lambda = 0.01
b3k_mse[b3k_mse >= mse_lambda] = mse_lambda
b6k_mse[b6k_mse >= mse_lambda] = mse_lambda
b9k_mse[b9k_mse >= mse_lambda] = mse_lambda
b12k_mse[b12k_mse >= mse_lambda] = mse_lambda
b3k_b6k_mse[b3k_b6k_mse >= mse_lambda] = mse_lambda
b3k_b9k_mse[b3k_b9k_mse >= mse_lambda] = mse_lambda

bin_size = 0.0001;
min_edge = 0;
max_edge = mse_lambda
N = (max_edge - min_edge) / bin_size;
Nplus1 = N + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)

n, bins, patches = plt.hist(x=b3k_mse, bins=bin_list, color={'b'},
                            alpha=0.7, rwidth=0.85, histtype='step')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels')
plt.show()
