from scipy.io import loadmat,savemat
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from multiple_shore_fits_main_path_calls import call_minimization_shore_on_data_with_basis_mse

shore_para_dict = {'radial_order':6,'zeta_guess':700.0,'lambdaN':1e-8,'lambdaL':1e-8}

fod_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_qspace.mat'
fod_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvals.bval'
fod_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_bvecs.bvec'

fod_shore_coeff, fod_basis, fod_mse = call_minimization_shore_on_data_with_basis_mse(fod_data_path, fod_bval_path, fod_bvec_path, 'new_q_fod', zeta_flag=1, log_flag=0, shore_para_dict=shore_para_dict)
print(fod_mse)

fod_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\fod_shore_basis\fod_shore_coeffs_r6.mat'
fod_shore_basis_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\fod_shore_basis\fod_basis_r6.mat'

fod_shore_path = os.path.normpath(fod_shore_path)
fod_shore_basis_path = os.path.normpath(fod_shore_basis_path)

savemat(fod_shore_path,mdict={'fod_shore_coeffs':fod_shore_coeff})
savemat(fod_shore_basis_path,mdict={'fod_shore_basis':fod_basis})
print('I think the code worked, FOD stuff is !')