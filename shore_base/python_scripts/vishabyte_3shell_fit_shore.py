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

bval_path = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvals_3shell.bval'
bvec_path = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvecs_3shell.bvec'
nifti_path = r'D:\MASI LAB WORK\Solid_Harmonics\vishabyte_concat_3shell.nii.gz'

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
radial_order = 8
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)



bval_path_5shell = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvals.bval'
bvec_path_5shell = r'D:\MASI LAB WORK\Solid_Harmonics\concat_bvecs.bvec'
nifti_path_5shell = r'D:\MASI LAB WORK\Solid_Harmonics\vishabyte_concat.nii.gz'
bvec_path_5shell = os.path.normpath(bvec_path_5shell)
bval_path_5shell = os.path.normpath(bval_path_5shell)
nifti_path_5shell = os.path.normpath(nifti_path_5shell)

nifti_5 = nib.load(nifti_path_5shell)
nifti_img_5shell = nifti_5.get_data()

bvals_5shell,bvecs_5shell = read_bvals_bvecs(bval_path_5shell, bvec_path_5shell)
gtab_5shell = gradient_table(bvals_5shell, bvecs_5shell)

asm5shell = ShoreModel(gtab_5shell, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)


asmfit = asm.fit(nifti_img)

asm5shell_fit = asm5shell.fit(nifti_img_5shell)

asm5shell_fit.fit_array = asmfit.fit_array

print('Shore Model Coeffs Generated ... \n')

print('Reconstructing Signal with new Shore\n')

signal_recon = asm5shell_fit.fitted_signal()
#signal_recon = asmfit.fitted_signal()

print('Signal Reconstructed \n')

#print('Calculating MSE per b-value')

savemat('vishabyte_intershell_predictions.mat',mdict={'predicted_signal': signal_recon})