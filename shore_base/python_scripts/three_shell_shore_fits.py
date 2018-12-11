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
from multiple_shore_fits_main_path_calls import call_minimization_shore_on_data

# Global SHORE Hyper-Parameters
radial_order = 6
zeta_guess = 700.0
lambdaN = 1e-8
lambdaL = 1e-8

shore_para_dict = {'radial_order':6,'zeta_guess':700.0,'lambdaN':1e-8,'lambdaL':1e-8}

# Three Shell Fitting (b3000, b6000, b9000) , (b3000, b9000, b12000) , (b6000, b9000, b12000)

b3k_b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_data.mat'
b3k_b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvals.bval'
b3k_b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvecs.bvec'

b3k_b6k_b9k_shore_coeff = call_minimization_shore_on_data(b3k_b6k_b9k_data_path, b3k_b6k_b9k_bval_path, b3k_b6k_b9k_bvec_path, 'b3k_b6k_b9k_data', 1, shore_para_dict)

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
