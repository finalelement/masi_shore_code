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
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bvec'

b3k_data_path = os.path.normpath(b3k_data_path)
b3k_bval_path = os.path.normpath(b3k_bval_path)
b3k_bvec_path = os.path.normpath(b3k_bvec_path)

bvals, bvecs = read_bvals_bvecs(b3k_bval_path, b3k_bvec_path)

b3k_bvals = bvals[101:201]
b3k_bvecs = bvecs[101:201,:]

gtab = gradient_table(b3k_bvals, b3k_bvecs)

b3k_data = loadmat(b3k_data_path)


b3k_data = b3k_data['new_fod_q_space']
one_vox = b3k_data[0:10,101:201]
print('data.shape (%d, %d)' % b3k_data.shape)

zeta = 0
#scale_val = 0.5
odf_stack = np.zeros((10,1,1,724))

for i in range(10):
    radial_order = 6
    zeta = zeta + 100
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


# Enables/disables interactive visualization
interactive = True
ren = window.Renderer()
sfu = actor.odf_slicer(odf_stack[0:10,None,0], sphere=sphere, colormap='plasma', scale=0.5, norm=True, radial_scale=True)
sfu.RotateX(-1)
sfu.RotateY(90)
sfu.display(z=0)
ren.add(sfu)
out_path_name = 'odfs_b12k' + str(zeta) + '.png'
window.record(ren, out_path=out_path_name, size=(200, 20))

#scale_val = scale_val + 0.5
if interactive:
    window.show(ren)


