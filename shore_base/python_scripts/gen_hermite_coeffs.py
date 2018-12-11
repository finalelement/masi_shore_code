from dipy.reconst.shore import ShoreModel
from dipy.viz import window, actor
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from scipy.io import loadmat,savemat
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import numpy as np

# Define Paths for the data and bvals and bvecs
data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_data.mat'
bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\ms_bvals.bval'
bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\ms_bvecs.bvec'

data_path = os.path.normpath(data_path)
bval_path = os.path.normpath(bval_path)
bvec_path = os.path.normpath(bvec_path)

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

print('Gradient Table Loaded and Ready for use')

data = loadmat(data_path)
actual_data = data['ms_data']

print('Data all loaded and ready for use')

print('data.shape (%d, %d)' % actual_data.shape)

radial_order = 8
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

asmfit = asm.fit(actual_data)

print('Shore Model Coeffs Generated ...')

print('Transforming object shape to save to a .mat file')

shore_input_matrix = np.zeros((57267,95))

for i in range(0,57267):

    temp_shore_coeffs = asmfit.fit_array[i]._shore_coef
    shore_input_matrix[i,:] = temp_shore_coeffs

    if (i%1000 == 0):
        print(i)

savemat('shore_coeffs_r8.mat',mdict={'input_shore': shore_input_matrix})



