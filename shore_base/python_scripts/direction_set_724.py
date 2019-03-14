import numpy as np
import os
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from scipy.io import loadmat,savemat

sphere = get_sphere('symmetric724')

grad_dirs = np.zeros((724,3))

print('Debug here')

grad_dirs[:,0] = sphere.x
grad_dirs[:,1] = sphere.y
grad_dirs[:,2] = sphere.z

save_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\direction_sets\point_set_724.mat'
save_path = os.path.normpath(save_path)

savemat(save_path,mdict={'gradients':grad_dirs})