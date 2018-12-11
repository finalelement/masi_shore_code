import os
import numpy as np
from scipy.io import loadmat,savemat
from deep_learning_models_training import build_base_network_sh_6th_acc, build_base_network_sh_8th_acc
from deep_learning_models_training import train_nn, save_estimate, build_base_network_sh_to_sh_8th_acc
from sklearn.model_selection import train_test_split, KFold

# Base Fail Results Path
base_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ipmi_results\failed_results_loop50_acc'

# Define Input Shore Path
b3k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_shore_coeffs.mat'
#b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_shore_coeffs.mat'

# Input SH Path
sh_input_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\input_matrix.mat'

# Define Output SH Path
sh_output_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\out_matrix.mat'

b3k_shore = loadmat(b3k_shore_path)
sh_output = loadmat(sh_output_path)
sh_input = loadmat(sh_input_path)

b3k_shore = b3k_shore['shore_coeffs']

# Split SH Output to 6th and 8th order matrices
sh_output = sh_output['out_matrix']
sh_output_6th = sh_output[:,:28]
sh_output_8th = sh_output[:,:45]

# SH Input Split up
sh_input = sh_input['input_matrix']
sh_input = sh_input[:,:45]

# Deep Learning Hyper-Parameters
iters = 200
bs_size = 10000

#################################################################
# We need a DL function that takes in X, Y, directory full path, iters, batch_size and thats it folks !

def dl_main_call_fail_models_sh_to_sh(X_final, Y_final, out_start_dir, iters=20, batch_size=10000):

    print('Debug here')

    dims = Y_final.shape
    model_D = build_base_network_sh_to_sh_8th_acc()

    indices = np.array(range(X_final.shape[0])) + 1

    #out_start_dir = os.path.join(dl_path, exp)
    if not os.path.exists(out_start_dir):
        os.makedirs(out_start_dir)

    seed1 = 46

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
        model_D = train_nn(model_D, X_train, y_train, out_dir_DNN, n_epoch=iters, val_size=0.2, batch_s=batch_size)

    print('Saving Predictions of Across Entire Data')
    test_path = os.path.join(out_start_dir, 'overall_result.mat')
    save_estimate(model_D, X_final, Y_final, test_path)





def dl_main_call_fail_models(X_final, Y_final, out_start_dir, iters=20, batch_size=10000):

    print('Debug here')

    dims = Y_final.shape
    if (dims[1] == 28):
        model_D = build_base_network_sh_6th_acc()
    elif (dims[1] == 45):
        model_D = build_base_network_sh_8th_acc()
    else:
        print('Y Dimensions do not match please verify !!')

    indices = np.array(range(X_final.shape[0])) + 1

    #out_start_dir = os.path.join(dl_path, exp)
    if not os.path.exists(out_start_dir):
        os.makedirs(out_start_dir)

    seed1 = 46

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
        model_D = train_nn(model_D, X_train, y_train, out_dir_DNN, n_epoch=iters, val_size=0.2, batch_s=batch_size)

    print('Saving Predictions of Across Entire Data')
    test_path = os.path.join(out_start_dir, 'overall_result.mat')
    save_estimate(model_D, X_final, Y_final, test_path)

###########################################################################

# SH 8th order -> SH 8th order
exp = 'sh_to_sh'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models_sh_to_sh(sh_input, sh_output_8th, exp_path, iters=iters, batch_size=bs_size)


# SHORE B3K -> SH 6th order
# exp = 'shore_b3k_sh_6th'
# exp_path = os.path.join(base_path,exp)
# print(exp_path)
# dl_main_call_fail_models(b3k_shore, sh_output_6th, exp_path, iters=iters, batch_size=bs_size)

# SHORE B3K -> SH 8th order
# exp = 'shore_b3k_sh_8th'
# exp_path = os.path.join(base_path,exp)
# print(exp_path)
# dl_main_call_fail_models(b3k_shore, sh_output_8th, exp_path, iters=iters, batch_size=bs_size)
