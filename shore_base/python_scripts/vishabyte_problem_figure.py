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

mask = nifti_img[:,:,:,0]
mask = np.round(mask)

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
plt.colorbar()
#plt.show()
#plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(mask[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.colorbar()
#plt.show()
plt.savefig('vishabyte_mid_axial_slices.png', bbox_inches='tight')
plt.clf()

print('Working on fitting SHORE to the Data \n')

zeta = 150

mse_vol_stack = np.zeros((dims[0],dims[1],dims[2],5))

for i in range(5):

    radial_order = 6
    lambdaN = 1e-8
    lambdaL = 1e-8
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

    asmfit = asm.fit(nifti_img, mask)

    print('Shore Model Coeffs Generated ... \n')

    print('Reconstructing Signal \n')
    signal_recon = asmfit.fitted_signal()

    mse_volume = np.zeros((dims[0],dims[1],dims[2]))

    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):

                if mask[x,y,z] == 1:
                    orig_sig = nifti_img[x,y,z,:]
                    recon_sig = signal_recon[x,y,z,:]
                    temp_error = np.mean((orig_sig - recon_sig) ** 2)
                    mse_volume[x,y,z] = temp_error

    mse_vol_stack[:,:,:,i] = mse_volume

    plt.imshow(mse_volume[:, :, axial_middle].T, cmap='gray', origin='lower', vmax=0.01, vmin=0)
    plt.colorbar()
    #plt.show()
    save_name = 'vish_mse_error_' + str(zeta) + '.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()
    zeta = zeta + 50

print('Lets Create Plots of MSE Values')

mse_1 = mse_vol_stack[:,:,:,0]
mse_2 = mse_vol_stack[:,:,:,1]
mse_3 = mse_vol_stack[:,:,:,2]
mse_4 = mse_vol_stack[:,:,:,3]

bool_mask = np.array(mask, dtype=bool)
mse_hist_1 = mse_1[bool_mask]
mse_hist_2 = mse_2[bool_mask]
mse_hist_3 = mse_3[bool_mask]
mse_hist_4 = mse_4[bool_mask]

mse_lambda = 0.04
mse_hist_1[mse_hist_1 >= mse_lambda] = mse_lambda
mse_hist_2[mse_hist_2 >= mse_lambda] = mse_lambda
mse_hist_3[mse_hist_3 >= mse_lambda] = mse_lambda
mse_hist_4[mse_hist_4 >= mse_lambda] = mse_lambda

bin_size = 0.0001;
min_edge = 0;
max_edge = mse_lambda
N = (max_edge - min_edge) / bin_size;
Nplus1 = N + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)

n, bins, patches = plt.hist(x=[mse_hist_1,mse_hist_2,mse_hist_3,mse_hist_4], bins=bin_list, color={'b', 'g', 'r', 'c'},
                            alpha=0.7, rwidth=0.85, histtype='step')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Voxels')
plt.show()
save_name = 'vish_mse_hist.png'
plt.savefig(save_name, bbox_inches='tight')
