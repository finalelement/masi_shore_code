from dipy.reconst.shore import ShoreModel
from dipy.viz import window, actor
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from scipy.io import loadmat,savemat
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

bval_path = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvals.bval'
bvec_path = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvecs.bvec'
nifti_path = r'D:\MASI LAB WORK\Solid_Harmonics\vishabyte_concat.nii.gz'

nifti_path = os.path.normpath(nifti_path)
bvec_path = os.path.normpath(bvec_path)
bval_path = os.path.normpath(bval_path)

bvals,bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)

nifti = nib.load(nifti_path)
nifti_img = nifti.get_data()

print('Nifti Information \n')
print('Dimensions \n')
print(nifti_img.shape)
dims = nifti_img.shape

print('Voxel Dimensions \n')
print(nifti.header.get_zooms()[:3])

print('Lets draw out some pictures of middle axial slices \n')
axial_middle = nifti_img.shape[2] // 2
#plt.figure('Showing the datasets')
#plt.subplots(1, 2, 1).set_axis_off()
plt.imshow(nifti_img[:, :, axial_middle, 30].T, cmap='gray', origin='lower', vmax=1, vmin=0)
plt.show()
#plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(nifti_img[:, :, axial_middle, 10].T, cmap='gray', origin='lower')
plt.savefig('vishabyte_mid_axial_slices.png', bbox_inches='tight')


print('Working on fitting SHORE to the Data \n')
radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

mask = nifti_img[:,:,:,0]
asmfit = asm.fit(nifti_img, mask)

print('Shore Model Coeffs Generated ... \n')

print('Reconstructing Signal \n')

signal_recon = asmfit.fitted_signal()

print('Signal Reconstructed \n')

shore_input_matrix = np.zeros((dims[0],dims[1],dims[2],50))

for i in range(0,dims[0]):
    for j in range (0,dims[1]):
        for k in range (0,dims[2]):
            if (mask[i,j,k] == 1):

                # Retrieve Shore Coeffs
                temp_shore = asmfit.fit_array[i,j,k]._shore_coef
                shore_input_matrix[i,j,k,:] = temp_shore

savemat('vishabyte_shore_coeffs_r6.mat',mdict={'vish_shore': shore_input_matrix})