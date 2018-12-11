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

shore_base_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore_zeta_guess'

shore_para_dict = {'radial_order':6,'zeta_guess':1780.0,'lambdaN':1e-8,'lambdaL':1e-8}

# Define Paths for the data and bvals and bvecs
# Single Shell fitting B3000, B6000, B9000 & B12000
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_data.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvals.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvecs.bvec'

b3k_shore_coeff = call_minimization_shore_on_data(b3k_data_path, b3k_bval_path, b3k_bvec_path, 'b3k_data', 0, shore_para_dict)

b3k_shore_path = shore_base_path + r'\b3k_shore_coeffs.mat'
savemat(b3k_shore_path,mdict={'shore_coeffs':b3k_shore_coeff})
print('I think the code worked, B3k is done !')

b6k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_data.mat'
b6k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_bvals.bval'
b6k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_bvecs.bvec'

b6k_shore_coeff = call_minimization_shore_on_data(b6k_data_path, b6k_bval_path, b6k_bvec_path, 'b6k_data', 0, shore_para_dict)

b6k_shore_path = shore_base_path + r'\b6k_shore_coeffs.mat'
savemat(b6k_shore_path,mdict={'shore_coeffs':b6k_shore_coeff})
print('B6k is done')

b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_data.mat'
b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_bvals.bval'
b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_bvecs.bvec'

b9k_shore_coeff = call_minimization_shore_on_data(b9k_data_path, b9k_bval_path, b9k_bvec_path, 'b9k_data', 0, shore_para_dict)

b9k_shore_path = shore_base_path + r'\b9k_shore_coeffs.mat'
savemat(b9k_shore_path,mdict={'shore_coeffs':b9k_shore_coeff})
print('B9k is done')

b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_data.mat'
b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvals.bval'
b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvecs.bvec'

b12k_shore_coeff = call_minimization_shore_on_data(b12k_data_path, b12k_bval_path, b12k_bvec_path, 'b12k_data', 0, shore_para_dict)

b12k_shore_path = shore_base_path + r'\b12k_shore_coeffs.mat'
savemat(b12k_shore_path,mdict={'shore_coeffs':b12k_shore_coeff})
print('B12k is done')

# Two Shell Fitting (B3000,B6000) , (B3000, B9000) , (B3000, B12000) , (B6000,B9000) , (B6000,B12000) , (B9000,B12000)
b3k_b6k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_data.mat'
b3k_b6k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvals.bval'
b3k_b6k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvecs.bvec'

b3k_b6k_shore_coeff = call_minimization_shore_on_data(b3k_b6k_data_path, b3k_b6k_bval_path, b3k_b6k_bvec_path, 'b3k_b6k_data', 0, shore_para_dict)

b3k_b6k_shore_path = shore_base_path + r'\b3k_b6k_shore_coeffs.mat'
savemat(b3k_b6k_shore_path,mdict={'shore_coeffs':b3k_b6k_shore_coeff})
print('B3k & B6k is done')

b3k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_data.mat'
b3k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_bvals.bval'
b3k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_bvecs.bvec'

b3k_b9k_shore_coeff = call_minimization_shore_on_data(b3k_b9k_data_path, b3k_b9k_bval_path, b3k_b9k_bvec_path, 'b3k_b9k_data', 0, shore_para_dict)

b3k_b9k_shore_path = shore_base_path + r'\b3k_b9k_shore_coeffs.mat'
savemat(b3k_b9k_shore_path,mdict={'shore_coeffs':b3k_b9k_shore_coeff})
print('B3k & B9k is done')


b3k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_data.mat'
b3k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_bvals.bval'
b3k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b12k_bvecs.bvec'

b3k_b12k_shore_coeff = call_minimization_shore_on_data(b3k_b12k_data_path, b3k_b12k_bval_path, b3k_b12k_bvec_path, 'b3k_b12k_data', 0, shore_para_dict)

b3k_b12k_shore_path = shore_base_path + r'\b3k_b12k_shore_coeffs.mat'
savemat(b3k_b12k_shore_path,mdict={'shore_coeffs':b3k_b12k_shore_coeff})
print('B3k & B12k is done')


b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_data.mat'
b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_bvals.bval'
b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_bvecs.bvec'

b6k_b9k_shore_coeff = call_minimization_shore_on_data(b6k_b9k_data_path, b6k_b9k_bval_path, b6k_b9k_bvec_path, 'b6k_b9k_data', 0, shore_para_dict)

b6k_b9k_shore_path = shore_base_path + r'\b6k_b9k_shore_coeffs.mat'
savemat(b6k_b9k_shore_path,mdict={'shore_coeffs':b6k_b9k_shore_coeff})
print('B6k & B9k is done')


b6k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_data.mat'
b6k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_bvals.bval'
b6k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b12k_bvecs.bvec'

b6k_b12k_shore_coeff = call_minimization_shore_on_data(b6k_b12k_data_path, b6k_b12k_bval_path, b6k_b12k_bvec_path, 'b6k_b12k_data', 0, shore_para_dict)

b6k_b12k_shore_path = shore_base_path + r'\b6k_b12k_shore_coeffs.mat'
savemat(b6k_b12k_shore_path,mdict={'shore_coeffs':b6k_b12k_shore_coeff})
print('B6k & B12k is done')



b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_data.mat'
b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_bvals.bval'
b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b9k_b12k_bvecs.bvec'

b9k_b12k_shore_coeff = call_minimization_shore_on_data(b9k_b12k_data_path, b9k_b12k_bval_path, b9k_b12k_bvec_path, 'b9k_b12k_data', 0, shore_para_dict)

b9k_b12k_shore_path = shore_base_path + r'\b9k_b12k_shore_coeffs.mat'
savemat(b9k_b12k_shore_path,mdict={'shore_coeffs':b9k_b12k_shore_coeff})
print('B9k & B12k is done')


# Three Shell Fitting (b3000, b6000, b9000) , (b3000, b9000, b12000) , (b6000, b9000, b12000)


b3k_b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_data.mat'
b3k_b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvals.bval'
b3k_b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvecs.bvec'

b3k_b6k_b9k_shore_coeff = call_minimization_shore_on_data(b3k_b6k_b9k_data_path, b3k_b6k_b9k_bval_path, b3k_b6k_b9k_bvec_path,'b3k_b6k_b9k_data', 0, shore_para_dict)

b3k_b6k_b9k_shore_path = shore_base_path + r'\b3k_b6k_b9k_shore_coeffs.mat'
savemat(b3k_b6k_b9k_shore_path,mdict={'shore_coeffs':b3k_b6k_b9k_shore_coeff})
print('B3k & B6k & B9K is done')


b3k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_data.mat'
b3k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_bvals.bval'
b3k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b9k_b12k_bvecs.bvec'

b3k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b3k_b9k_b12k_data_path, b3k_b9k_b12k_bval_path, b3k_b9k_b12k_bvec_path, 'b3k_b9k_b12k_data', 0, shore_para_dict)

b3k_b9k_b12k_shore_path = shore_base_path + r'\b3k_b9k_b12k_shore_coeffs.mat'
savemat(b3k_b9k_b12k_shore_path,mdict={'shore_coeffs':b3k_b9k_b12k_shore_coeff})
print('B3k & B9k & B12K is done')


b6k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_data.mat'
b6k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_bvals.bval'
b6k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b6k_b9k_b12k_bvecs.bvec'

b6k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b6k_b9k_b12k_data_path, b6k_b9k_b12k_bval_path, b6k_b9k_b12k_bvec_path, 'b6k_b9k_b12k_data', 0, shore_para_dict)

b6k_b9k_b12k_shore_path = shore_base_path + r'\b6k_b9k_b12k_shore_coeffs.mat'
savemat(b6k_b9k_b12k_shore_path,mdict={'shore_coeffs':b6k_b9k_b12k_shore_coeff})
print('B6k & B9k & B12K is done')


b3k_b6k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_data.mat'
b3k_b6k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvals.bval'
b3k_b6k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvecs.bvec'

b3k_b6k_b9k_b12k_shore_coeff = call_minimization_shore_on_data(b3k_b6k_b9k_b12k_data_path, b3k_b6k_b9k_b12k_bval_path, b3k_b6k_b9k_b12k_bvec_path, 'b3k_b6k_b9k_b12k_data', 0, shore_para_dict)

b3k_b6k_b9k_b12k_shore_path = shore_base_path + r'\b3k_b6k_b9k_b12k_shore_coeffs.mat'
savemat(b3k_b6k_b9k_b12k_shore_path,mdict={'shore_coeffs':b3k_b6k_b9k_b12k_shore_coeff})
print('B3k & B6k & B9k & B12K is done')

#b3k_data_path = os.path.normpath(b3k_data_path)
#b3k_bval_path = os.path.normpath(b3k_bval_path)
#b3k_bvec_path = os.path.normpath(b3k_bvec_path)

#b3k_bvals, b3k_bvecs = read_bvals_bvecs(b3k_bval_path, b3k_bvec_path)
#b3k_gtab = gradient_table(b3k_bvals, b3k_bvecs)

#print('b3k Gradient Table Loaded and Ready for use')

#b3k_data = loadmat(b3k_data_path)
#b3k_data = b3k_data['b3k_data']

#b3k_zeta = returns_best_zeta(b3k_data, radial_order, b3k_gtab, zeta_guess)

#print('Zeta Estimated for B3000 Just')

#b3k_shore_coeff, b3k_shore_preds = return_shore_coeffs_preds(b3k_data, radial_order, b3k_gtab, b3k_zeta)

#print('Plot MSE of b3k Shore')

#b3k_title = 'B3000 Error between Shore Predictions & Original Data'
#b3k_mse_vector = return_mse_vector_coeffs_preds(b3k_data, b3k_shore_preds, b3k_title)

####################################################################################

# Define Paths for the data and bvals and bvecs

