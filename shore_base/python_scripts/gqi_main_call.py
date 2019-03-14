import os
import numpy as np
from gqi_functions_vish import reconst_gqi_fodf_sh_coeffs, reconst_gqi_fodf_return_sh_coeffs
from gqi_functions_vish import fiber_nav_sh_to_masi_sh_order_8th
from scipy.io import loadmat,savemat

base_save_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\gqi_combinations'
base_save_path = os.path.normpath(base_save_path)

# Single Shell fitting B3000, B6000, B9000 & B12000
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_data.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvals.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvecs.bvec'

#b3k_gq_odfs = reconst_gqi_fodf_sh_coeffs(b3k_bval_path,b3k_bvec_path,b3k_data_path,'b3k_data')
b3k_gq_odfs_sh = reconst_gqi_fodf_sh_coeffs(b3k_bval_path,b3k_bvec_path,b3k_data_path,'b3k_data')

#b3k_gq_odfs_sh = fiber_nav_sh_to_masi_sh_order_8th(b3k_gq_odfs_sh)

b3k_odf_path = os.path.join(base_save_path,'b3k_odf_sh.mat')
savemat(b3k_odf_path,mdict={'b3k_gqi_odf_sh':b3k_gq_odfs_sh})
