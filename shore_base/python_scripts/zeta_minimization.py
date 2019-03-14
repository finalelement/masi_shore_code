import numpy as np
from dipy.reconst.shore import shore_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
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


def eval_minimized_zeta(D, n, gtab, scale):
    lambdaN = 1e-8
    lambdaL = 1e-8
    radial_order = n

    Lshore = l_shore(radial_order)
    Nshore = n_shore(radial_order)

    M = shore_matrix(n, scale, gtab, 1. / (4 * np.pi ** 2))
    MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
    shorecoefs = np.dot(D, MpseudoInv.T)

    # shorecoefs = np.dot(D, pinv(Mshore).T)
    shorepred = np.dot(shorecoefs, M.T)
    return np.linalg.norm(D - shorepred) ** 2


def returns_best_zeta(actual_data, radial_deg, gtab, init_guess):
    zeta = minimize(lambda x: eval_minimized_zeta(actual_data[:5000, :], radial_deg, gtab, x), init_guess)['x']
    print(zeta)
    return zeta

def return_shore_basis(n,gtab,scale):
    lambdaN = 1e-8
    lambdaL = 1e-8
    radial_order = n

    Lshore = l_shore(radial_order)
    Nshore = n_shore(radial_order)

    M = shore_matrix(n, scale, gtab, 1. / (4 * np.pi ** 2))
    return M

def return_shore_coeffs_preds(D, n, gtab, scale):

    lambdaN = 1e-8
    lambdaL = 1e-8
    radial_order = n

    Lshore = l_shore(radial_order)
    Nshore = n_shore(radial_order)

    M = shore_matrix(n, scale, gtab, 1. / (4 * np.pi ** 2))
    MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN * Nshore + lambdaL * Lshore), M.T)
    shore_coeffs = np.dot(D, MpseudoInv.T)

    shore_preds = np.dot(shore_coeffs, M.T)

    return shore_coeffs, shore_preds

def return_mse_vector_coeffs_preds(b3k_data, b3k_shore_preds, plot_title):

    sample_size = len(b3k_data)
    mse_vector = np.zeros((sample_size,1))

    for i in range(0, sample_size):
        orig_sig = b3k_data[i, :]
        recon_sig = b3k_shore_preds[i, :]

        temp_mse = np.mean((orig_sig - recon_sig) ** 2)
        mse_vector[i] = temp_mse

    # Parameter to control the outlier MSE values

    mse_lambda = 0.01
    mse_vector[mse_vector >= mse_lambda] = mse_lambda

    bin_size = 0.0001;
    min_edge = 0;
    max_edge = mse_lambda
    N = (max_edge - min_edge) / bin_size;
    Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)

    n, bins, patches = plt.hist(x=mse_vector, bins=bin_list, color='#0504aa',
                                alpha=0.7, rwidth=0.85, histtype='bar')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Number of Voxels - Total 57000')
    plt.title(plot_title)
    maxfreq = n.max()
    plt.show()


    return mse_vector


def return_mse_vector_coeffs_preds_log(b3k_data, b3k_shore_preds, plot_title):

    sample_size = len(b3k_data)
    mse_vector = np.zeros((sample_size,1))

    for i in range(0, sample_size):
        orig_sig = b3k_data[i, :]
        orig_sig = np.exp(orig_sig)
        recon_sig = b3k_shore_preds[i, :]
        recon_sig = np.exp(recon_sig)

        temp_mse = np.mean((orig_sig - recon_sig) ** 2)
        mse_vector[i] = temp_mse

    # Parameter to control the outlier MSE values

    mse_lambda = 0.01
    mse_vector[mse_vector >= mse_lambda] = mse_lambda

    bin_size = 0.0001;
    min_edge = 0;
    max_edge = mse_lambda
    N = (max_edge - min_edge) / bin_size;
    Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)

    n, bins, patches = plt.hist(x=mse_vector, bins=bin_list, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Number of Voxels - Total 57000')
    plt.title(plot_title)
    maxfreq = n.max()
    plt.show()


    return mse_vector

def return_overall_mse_vector_coeffs_preds(b3k_data, b3k_shore_preds, plot_title):

    return np.mean(np.abs(b3k_data - b3k_shore_preds))
