import os
import numpy as np
from scipy.io import loadmat,savemat
from zeta_minimization import returns_best_zeta, return_shore_coeffs_preds, return_mse_vector_coeffs_preds, return_mse_vector_coeffs_preds_log
from multiple_shore_fits_main_path_calls import call_minimization_shore_on_data_with_basis

fod_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_qspace.mat'
fod_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvals.bval'
fod_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvecs.bvec'

base_save_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri'
#fod_path = os.path.normpath(fod_path)
#fod_bval_path = os.path.normpath(fod_bval_path)
#fod_bvec_path = os.path.normpath(fod_bvec_path)

shore_para_dict = {'radial_order':6,'zeta_guess':700.0,'lambdaN':1e-8,'lambdaL':1e-8}

fod_shore_coeff, fod_real_basis = call_minimization_shore_on_data_with_basis(fod_path, fod_bval_path, fod_bvec_path, 'new_q_fod', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)

shore_coeff_path = base_save_path + r'\fod_shore_single_shell.mat'
savemat(shore_coeff_path,mdict={'fod_shore_coeffs':fod_shore_coeff})

shore_basis_path = base_save_path + r'\fod_shore_single_shell_basis_r6.mat'
savemat(shore_basis_path,mdict={'fod_shore_basis':fod_real_basis})

print('FOD Single Shell Shore Coefficients and Basis Saved')

print('Create Log Version of the same')

ln_fod_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ln_fod_qspace.mat'
ln_bvals_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvals_no_b0.bval'
ln_bvecs_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvecs_no_b0.bvec'

ln_fod_shore_coeff, fod_log_basis = call_minimization_shore_on_data_with_basis(ln_fod_path, fod_bval_path, fod_bvec_path, 'ln_new_q_fod', zeta_flag=1, log_flag=1, shore_para_dict=shore_para_dict)

log_shore_coeff_path = base_save_path + r'\log_fod_shore_single_shell.mat'
savemat(log_shore_coeff_path,mdict={'ln_fod_shore_coeffs':ln_fod_shore_coeff})

log_shore_basis_path = base_save_path + r'\log_fod_shore_single_shell_basis_r6.mat'
savemat(log_shore_basis_path,mdict={'log_fod_shore_basis':fod_log_basis})


print('All Said and done whats going on ?')