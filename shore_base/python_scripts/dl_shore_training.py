import os
import numpy as np
from scipy.io import loadmat,savemat
import random
from sklearn.model_selection import train_test_split, KFold
from deep_learning_models_training import build_base_network, train_nn
from deep_learning_models_training import save_test_set_prediction
# Base Deep Learning Path
dl_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ipmi_results\zeta_optimized_fod_single_shell'

# Output Data FOD Paths
# fod_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\shore_coeffs_decayed_fod_r6.mat'
fod_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_shore_single_shell.mat'

# Training Data Paths
# We are training on B3000, B9000, B12000, (B3K,B9K) , (B3K,B12K) , (B9K,B12K) , (B3K, B9K, B12K)

b3k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_shore_coeffs.mat'
b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_shore_coeffs.mat'
b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b12k_shore_coeffs.mat'
b3k_b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b9k_shore_coeffs.mat'
b3k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b12k_shore_coeffs.mat'
b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_b12k_shore_coeffs.mat'
b3k_b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b9k_b12k_shore_coeffs.mat'

# Testing Data
b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_shore_coeffs.mat'
b3k_b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_shore_coeffs.mat'
b3k_b6k_b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_b9k_shore_coeffs.mat'
b3k_b6k_b9k_b12k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_b6k_b9k_b12k_shore_coeffs.mat'

# Loading Training Data
b3k_shore = loadmat(b3k_shore_path)
b9k_shore = loadmat(b9k_shore_path)
b12k_shore = loadmat(b12k_shore_path)
b3k_b9k_shore = loadmat(b3k_b9k_shore_path)
b3k_b12k_shore = loadmat(b3k_b12k_shore_path)
b9k_b12k_shore = loadmat(b9k_b12k_shore_path)
b3k_b9k_b12k_shore = loadmat(b3k_b9k_b12k_shore_path)

# Loading Testing Data
b6k_shore = loadmat(b6k_shore_path)
b3k_b6k_shore = loadmat(b3k_b6k_shore_path)
b3k_b6k_b9k_shore = loadmat(b3k_b6k_b9k_shore_path)
b3k_b6k_b9k_b12k_shore = loadmat(b3k_b6k_b9k_b12k_shore_path)

print('All Data Loaded')

b3k_shore = b3k_shore['shore_coeffs']
b9k_shore = b9k_shore['shore_coeffs']
b12k_shore = b12k_shore['shore_coeffs']
b3k_b9k_shore = b3k_b9k_shore['shore_coeffs']
b3k_b12k_shore = b3k_b12k_shore['shore_coeffs']
b9k_b12k_shore = b9k_b12k_shore['shore_coeffs']
b3k_b9k_b12k_shore = b3k_b9k_b12k_shore['shore_coeffs']

b6k_shore = b6k_shore['shore_coeffs']
b3k_b6k_shore = b3k_b6k_shore['shore_coeffs']
b3k_b6k_b9k_shore = b3k_b6k_b9k_shore['shore_coeffs']
b3k_b6k_b9k_b12k_shore = b3k_b6k_b9k_b12k_shore['shore_coeffs']


print('Data Extracted')

# Load Output Data
fod_shore = loadmat(fod_data_path)
#fod_shore = fod_shore['output_shore']
fod_shore = fod_shore['fod_shore_coeffs']

print('FOD Loaded')

# Define the indices for slicing Training and Testing Data
t_s_start_index = [0,7272,14544,21816,29088,36360,43632,50904]
t_s_end_index = [7272,14544,21816,29088,36360,43632,50904,57267]

for i in range(len(t_s_start_index)):

    print('Indices: ',t_s_start_index[i],' to ',t_s_end_index[i],' are used for Test Set!')

    deletion_indices = np.arange(t_s_start_index[i],t_s_end_index[i])
    # Form the Training Input Set
    t_b3k_shore = np.delete(b3k_shore, deletion_indices, axis=0)
    t_b9k_shore = np.delete(b9k_shore, deletion_indices, axis=0)
    t_b12k_shore = np.delete(b12k_shore, deletion_indices, axis=0)
    t_b3k_b9k_shore = np.delete(b3k_b9k_shore, deletion_indices, axis=0)
    t_b3k_b12k_shore = np.delete(b3k_b12k_shore, deletion_indices, axis=0)
    t_b9k_b12k_shore = np.delete(b9k_b12k_shore, deletion_indices, axis=0)
    t_b3k_b9k_b12_kshore = np.delete(b3k_b9k_b12k_shore, deletion_indices, axis=0)

    X = np.concatenate((t_b3k_shore, t_b9k_shore, t_b12k_shore, t_b3k_b9k_shore, t_b3k_b12k_shore, t_b9k_b12k_shore, t_b3k_b9k_b12_kshore), axis=0)

    # Form Training Output Set
    t_fod_shore = np.delete(fod_shore, deletion_indices, axis=0)
    Y = np.concatenate((t_fod_shore, t_fod_shore, t_fod_shore, t_fod_shore, t_fod_shore, t_fod_shore, t_fod_shore), axis=0)

    # Let's Randomize our Training Input/Output Data
    sample_length = len(X)
    rand_indices = random.sample(range(sample_length), sample_length)

    X_final = X[rand_indices,:]
    Y_final = Y[rand_indices,:]

    # Extract the same test indices from testing sets
    test_b6k_shore_input = b6k_shore[t_s_start_index[i]:t_s_end_index[i], :]
    test_b3k_b6k_shore_input = b3k_b6k_shore[t_s_start_index[i]:t_s_end_index[i], :]
    test_b3k_b6k_b9k_shore_input = b3k_b6k_b9k_shore[t_s_start_index[i]:t_s_end_index[i], :]
    test_b3k_b6k_b9k_b12k_shore_input = b3k_b6k_b9k_b12k_shore[t_s_start_index[i]:t_s_end_index[i], :]

    # Let's Deep Learn on the Dataset
    model_D = build_base_network()

    iters = 2
    indices = np.array(range(X_final.shape[0])) + 1

    exp = 'f_' + str(i)
    out_start_dir = os.path.join(dl_path, exp)
    if not os.path.exists(out_start_dir):
        os.makedirs(out_start_dir)

    seed1 = 46
    seed2 = 23

    kf = KFold(n_splits=5, random_state=seed1, shuffle=True, )

    fold_num = 0

    for train, test in kf.split(X_final):
        # Set up training / testing data
        fold_num += 1
        X_train = X_final[train, :]
        y_train = Y_final[train, :]
        X_test = X_final[test, :]
        y_test = Y_final[test, :]
        indices_train = indices[train]
        indices_test = indices[test]

        # Deep NN
        print("Training DNN with %d iterations, fold %d" % (iters, fold_num))
        out_dir_DNN = os.path.join(out_start_dir, str(fold_num))
        model_D = train_nn(model_D, X_train, y_train, out_dir_DNN, n_epoch=iters, val_size=0.1)

    print('Saving Predictions of Test Set B6000')
    b6k_test_path = os.path.join(out_start_dir, str('Hist_Blind_Test'), 'b6k_test_result.mat')
    save_test_set_prediction(model_D, b6k_test_path, test_b6k_shore_input)

    print('Saving Predictions of Test Set B3000, B6000')
    b3k_b6k_test_path = os.path.join(out_start_dir, str('Hist_Blind_Test'), 'b3k_b6k_test_result.mat')
    save_test_set_prediction(model_D, b3k_b6k_test_path, test_b3k_b6k_shore_input)

    print('Saving Predictions of Test Set B3000, B6000, B9000')
    b3k_b6k_b9k_test_path = os.path.join(out_start_dir, str('Hist_Blind_Test'), 'b3k_b6k_b9k_test_result.mat')
    save_test_set_prediction(model_D, b3k_b6k_b9k_test_path, test_b3k_b6k_b9k_shore_input)

    print('Saving Predictions of Test Set B3000, B6000, B9000, B12000')
    b3k_b6k_b9k_b12k_test_path = os.path.join(out_start_dir, str('Hist_Blind_Test'), 'b3k_b6k_b9k_b12k_test_result.mat')
    save_test_set_prediction(model_D, b3k_b6k_b9k_b12k_test_path, test_b3k_b6k_b9k_b12k_shore_input)

    print('Loop finished')