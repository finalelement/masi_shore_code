from dipy.reconst import mapmri
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

small_delta = 0.008 #in seconds
big_delta =  0.02 #in seconds
bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta, small_delta=small_delta)

print('Gradient Table Loaded and Ready for use \n')

data = loadmat(data_path)
actual_data = data['ms_data']

print('Data all loaded and ready for use \n')
print('data.shape (%d, %d)' % actual_data.shape)

radial_order = 6
map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                          laplacian_regularization=True,
                                          laplacian_weighting='GCV',
                                          positivity_constraint=True)

mapfit_both_aniso = map_model_both_aniso.fit(actual_data)

print('MapMri signal coefficients estimated')