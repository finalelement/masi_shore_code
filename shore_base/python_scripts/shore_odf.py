from dipy.reconst.shore import ShoreModel
#from dipy.viz import window, actor
#from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from scipy.io import loadmat,savemat
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from dipy.data import get_sphere
from dipy.viz import window, actor

# Define Paths for the data and bvals and bvecs
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_data.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvals.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b12k_bvecs.bvec'

b3k_data_path = os.path.normpath(b3k_data_path)
b3k_bval_path = os.path.normpath(b3k_bval_path)
b3k_bvec_path = os.path.normpath(b3k_bvec_path)

bvals, bvecs = read_bvals_bvecs(b3k_bval_path, b3k_bvec_path)
gtab = gradient_table(bvals,bvecs)

b3k_data = loadmat(b3k_data_path)
b3k_data = b3k_data['b12k_data']
one_vox = b3k_data[0:10,:]
print('data.shape (%d, %d)' % b3k_data.shape)

zeta = 100
#scale_val = 0.5
num_vox = 5
odf_stack = np.zeros((num_vox,1,1,724))

for i in range(num_vox):
    radial_order = 6
    lambdaN = 1e-8
    lambdaL = 1e-8
    print('Zeta Value: \n')
    print(zeta)
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

    asmfit = asm.fit(one_vox)

    sphere = get_sphere('symmetric724')

    odf = asmfit.odf(sphere)
    odf = np.reshape(odf,[10,1,1,724])

    odf_stack[i,:,:,:] = odf[0,:,:,:]
    print('odf.shape (%d, %d, %d, %d)' % odf.shape)
    zeta = zeta + 200

# Enables/disables interactive visualization
interactive = True
ren = window.Renderer()
sfu = actor.odf_slicer(odf_stack[0:5,None,0], sphere=sphere, colormap='plasma', scale=4, norm=False, radial_scale=True)
sfu.RotateX(-1)
sfu.RotateY(90)
sfu.display(z=0)
ren.add(sfu)
out_path_name = 'odfs_b6k' + str(zeta) + '.png'
window.record(ren, out_path=out_path_name, size=(200, 20))

#scale_val = scale_val + 0.5
if interactive:
    window.show(ren)


