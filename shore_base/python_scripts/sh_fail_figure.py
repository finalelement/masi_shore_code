import os
import numpy as np
from scipy.io import loadmat,savemat
from deep_learning_models_training import build_base_network_sh_6th, build_base_network_sh_8th, build_base_network
from deep_learning_models_training import train_nn, save_estimate
from sklearn.model_selection import train_test_split, KFold

# Base Fail Results Path
base_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ipmi_results\failed_results_loop50'

# Define Input Shore Path
b3k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b3k_shore_coeffs.mat'
b9k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b9k_shore_coeffs.mat'
#b6k_shore_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_shore\b6k_shore_coeffs.mat'

# Define Output SH Path
sh_output_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\out_matrix.mat'
shore_output_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\fod_shore_single_shell.mat'

b3k_shore = loadmat(b3k_shore_path)
b9k_shore = loadmat(b9k_shore_path)

sh_output = loadmat(sh_output_path)

b3k_shore = b3k_shore['shore_coeffs']
b9k_shore = b9k_shore['shore_coeffs']

# SHORE Output
shore_output = loadmat(shore_output_path)
shore_output = shore_output['fod_shore_coeffs']

# Split SH Output to 6th and 8th order matrices
sh_output = sh_output['out_matrix']
sh_output_6th = sh_output[:,:28]
sh_output_8th = sh_output[:,:45]

# Deep Learning Hyper-Parameters
iters = 100
bs_size = 10000


#################################################################
# We need a DL function that takes in X, Y, directory full path, iters, batch_size and thats it folks !

def dl_main_call_fail_models(X_final, Y_final, out_start_dir, iters=20, batch_size=10000):

    print('Debug here')

    dims = Y_final.shape
    if (dims[1] == 28):
        model_D = build_base_network_sh_6th()
    elif (dims[1] == 45):
        model_D = build_base_network_sh_8th()
    elif (dims[1] == 50):
        model_D = build_base_network()
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


# SHORE B3K -> SH 6th order
exp = 'shore_b3k_sh_6th'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b3k_shore, sh_output_6th, exp_path, iters=iters, batch_size=bs_size)

# SHORE B3K -> SH 8th order
exp = 'shore_b3k_sh_8th'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b3k_shore, sh_output_8th, exp_path, iters=iters, batch_size=bs_size)

# SHORE B9K -> SH 6th order
exp = 'shore_b9k_sh_6th'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b9k_shore, sh_output_6th, exp_path, iters=iters, batch_size=bs_size)

# SHORE B9K -> SH 8th order
exp = 'shore_b9k_sh_8th'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b9k_shore, sh_output_8th, exp_path, iters=iters, batch_size=bs_size)


# SHORE B3K -> SHORE FOD
exp = 'shore_b3k_shore_fod'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b3k_shore, shore_output, exp_path, iters=iters, batch_size=bs_size)


# SHORE B9K -> SHORE FOD
exp = 'shore_b9k_shore_fod'
exp_path = os.path.join(base_path,exp)
print(exp_path)
dl_main_call_fail_models(b9k_shore, shore_output, exp_path, iters=iters, batch_size=bs_size)




