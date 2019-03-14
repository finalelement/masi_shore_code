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
from zeta_minimization import returns_best_zeta, return_shore_coeffs_preds
from zeta_minimization import return_mse_vector_coeffs_preds, return_mse_vector_coeffs_preds_log, return_shore_basis
from zeta_minimization import return_overall_mse_vector_coeffs_preds

def call_minimization_shore_on_data(data_path, bval_path, bvec_path, data_var, zeta_flag, log_flag, shore_para_dict):

    print('Data in Process is \n')
    print(data_var)
    # Unpack Shore Parameters
    radial_order = shore_para_dict['radial_order']
    zeta_guess = shore_para_dict['zeta_guess']
    lambdaN = shore_para_dict['lambdaN']
    lambdaL = shore_para_dict['lambdaL']

    data_path = os.path.normpath(data_path)
    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals,bvecs)

    print('Gradient Table Loaded and Ready for use \n')

    data = loadmat(data_path)
    data = data[data_var]

    if zeta_flag==1:
        best_zeta = returns_best_zeta(data, radial_order, gtab, zeta_guess)

        print('Zeta Estimated \n')
        #print(best_zeta)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, best_zeta)

    else:

        print('Zeta was not Estimated because the flag was not set')
        print(zeta_guess)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, zeta_guess)

    print('Calculating and Plotting MSE')

    if zeta_flag==0:
        best_zeta=zeta_guess

    plot_title = data_var + ' Error b/w Shore & Orig Data, Zeta: ' + str(best_zeta)

    if log_flag==0:
        mse_vector = return_mse_vector_coeffs_preds(data, shore_preds, plot_title)

    if log_flag==1:
        plot_title = data_var + ' Error b/w Shore & Orig Data, Log Enforced, Zeta: ' + str(best_zeta)
        mse_vector = return_mse_vector_coeffs_preds_log(data, shore_preds, plot_title)

    return shore_coeff

def call_minimization_shore_on_data_with_basis(data_path, bval_path, bvec_path, data_var, zeta_flag, log_flag, shore_para_dict):

    print('Data in Process is \n')
    print(data_var)
    # Unpack Shore Parameters
    radial_order = shore_para_dict['radial_order']
    zeta_guess = shore_para_dict['zeta_guess']
    lambdaN = shore_para_dict['lambdaN']
    lambdaL = shore_para_dict['lambdaL']

    data_path = os.path.normpath(data_path)
    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs)

    print('Gradient Table Loaded and Ready for use \n')

    data = loadmat(data_path)
    data = data[data_var]

    if zeta_flag == 1:
        best_zeta = returns_best_zeta(data, radial_order, gtab, zeta_guess)

        print('Zeta Estimated \n')
        # print(best_zeta)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, best_zeta)

    else:

        print('Zeta was not Estimated because the flag was not set')
        print(zeta_guess)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, zeta_guess)

    print('Calculating and Plotting MSE')

    if zeta_flag == 0:
        best_zeta = zeta_guess

    plot_title = data_var + ' Error b/w Shore & Orig Data, Zeta: ' + str(best_zeta)

    if log_flag == 0:
        mse_vector = return_mse_vector_coeffs_preds(data, shore_preds, plot_title)

    if log_flag == 1:
        plot_title = data_var + ' Error b/w Shore & Orig Data, Log Enforced, Zeta: ' + str(best_zeta)
        mse_vector = return_mse_vector_coeffs_preds_log(data, shore_preds, plot_title)

    print('Best Zeta Fed for Basis set ' + str(best_zeta))
    shore_basis = return_shore_basis(radial_order, gtab, scale=best_zeta)

    return shore_coeff, shore_basis

def call_minimization_shore_on_data_with_basis_mse(data_path, bval_path, bvec_path, data_var, zeta_flag, log_flag, shore_para_dict):

    print('Data in Process is \n')
    print(data_var)
    # Unpack Shore Parameters
    radial_order = shore_para_dict['radial_order']
    zeta_guess = shore_para_dict['zeta_guess']
    lambdaN = shore_para_dict['lambdaN']
    lambdaL = shore_para_dict['lambdaL']

    data_path = os.path.normpath(data_path)
    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

    # DELETE THIS LINE ONCE YOU ARE DONE PLAYING WITH FOD ERROR
    #bvals[1:] = 3000
    gtab = gradient_table(bvals, bvecs)

    print('Gradient Table Loaded and Ready for use \n')

    data = loadmat(data_path)
    data = data[data_var]

    if zeta_flag == 1:
        best_zeta = returns_best_zeta(data, radial_order, gtab, zeta_guess)

        print('Zeta Estimated \n')
        # print(best_zeta)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, best_zeta)

    else:

        print('Zeta was not Estimated because the flag was not set')
        print(zeta_guess)
        shore_coeff, shore_preds = return_shore_coeffs_preds(data, radial_order, gtab, zeta_guess)

    print('Calculating and Plotting MSE')

    if zeta_flag == 0:
        best_zeta = zeta_guess

    plot_title = data_var + ' Error b/w Shore & Orig Data, Zeta: ' + str(best_zeta)

    if log_flag == 0:
        #mse_vector = return_mse_vector_coeffs_preds(data, shore_preds, plot_title)
        mse_vector = return_mse_vector_coeffs_preds(data, shore_preds, plot_title)

    if log_flag == 1:
        plot_title = data_var + ' Error b/w Shore & Orig Data, Log Enforced, Zeta: ' + str(best_zeta)
        mse_vector = return_mse_vector_coeffs_preds_log(data, shore_preds, plot_title)

    print('Best Zeta Fed for Basis set ' + str(best_zeta))
    shore_basis = return_shore_basis(radial_order, gtab, scale=best_zeta)

    return shore_coeff, shore_basis, mse_vector