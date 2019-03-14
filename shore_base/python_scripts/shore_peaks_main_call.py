import os
import numpy as np
from shore_functions_vish import reconstruct_shore_fodf
from gqi_functions_vish import fiber_nav_sh_to_masi_sh_order_8th
from scipy.io import loadmat,savemat

base_save_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\shore_peaks'
base_save_path = os.path.normpath(base_save_path)

# Single Shell fitting B3000, B6000, B9000 & B12000
b3k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_data.mat'
b3k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvals.bval'
b3k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_bvecs.bvec'

#b3k_gq_odfs = reconst_gqi_fodf_sh_coeffs(b3k_bval_path,b3k_bvec_path,b3k_data_path,'b3k_data')
b3k_shore_odfs_sh = reconstruct_shore_fodf(b3k_bval_path,b3k_bvec_path,b3k_data_path,'b3k_data',zeta_val=700.0)
b3k_shore_odfs_sh = fiber_nav_sh_to_masi_sh_order_8th(b3k_shore_odfs_sh)
b3k_odf_path = os.path.join(base_save_path,'b3k_shore_fodf_sh.mat')
savemat(b3k_odf_path,mdict={'b3k_gqi_odf_sh':b3k_shore_odfs_sh})

####################################################

# Two Shell Fitting (B3000,B6000) , (B3000, B9000) , (B3000, B12000) , (B6000,B9000) , (B6000,B12000) , (B9000,B12000)
b3k_b6k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_data.mat'
b3k_b6k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvals.bval'
b3k_b6k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_bvecs.bvec'

b3k_b6k_shore_odfs_sh = reconstruct_shore_fodf(b3k_b6k_bval_path,b3k_b6k_bvec_path,b3k_b6k_data_path,'b3k_b6k_data',zeta_val=802.0)
b3k_b6k_shore_odfs_sh = fiber_nav_sh_to_masi_sh_order_8th(b3k_b6k_shore_odfs_sh)
b3k_b6k_odf_path = os.path.join(base_save_path,'b3k_b6k_shore_fodf_sh.mat')
savemat(b3k_b6k_odf_path,mdict={'b3k_b6k_gqi_odf_sh':b3k_b6k_shore_odfs_sh})

#####################################################

# Three Shell Fitting (b3000, b6000, b9000) , (b3000, b9000, b12000) , (b6000, b9000, b12000)
b3k_b6k_b9k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_data.mat'
b3k_b6k_b9k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvals.bval'
b3k_b6k_b9k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_bvecs.bvec'

b3k_b6k_b9k_shore_odfs_sh = reconstruct_shore_fodf(b3k_b6k_b9k_bval_path,b3k_b6k_b9k_bvec_path,b3k_b6k_b9k_data_path,'b3k_b6k_b9k_data',zeta_val=700.0)
b3k_b6k_b9k_shore_odfs_sh = fiber_nav_sh_to_masi_sh_order_8th(b3k_b6k_b9k_shore_odfs_sh)
b3k_b6k_b9k_odf_path = os.path.join(base_save_path,'b3k_b6k_b9k_shore_fodf_sh.mat')
savemat(b3k_b6k_b9k_odf_path,mdict={'b3k_b6k_b9k_gqi_odf_sh':b3k_b6k_b9k_shore_odfs_sh})


#####################################################

b3k_b6k_b9k_b12k_data_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_data.mat'
b3k_b6k_b9k_b12k_bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvals.bval'
b3k_b6k_b9k_b12k_bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\multi_shell_combinations\b3k_b6k_b9k_b12k_bvecs.bvec'

b3k_b6k_b9k_b12k_shore_odfs_sh = reconstruct_shore_fodf(b3k_b6k_b9k_b12k_bval_path,b3k_b6k_b9k_b12k_bvec_path,b3k_b6k_b9k_b12k_data_path,'b3k_b6k_b9k_b12k_data',zeta_val=700.0)
b3k_b6k_b9k_b12k_shore_odfs_sh = fiber_nav_sh_to_masi_sh_order_8th(b3k_b6k_b9k_b12k_shore_odfs_sh)
b3k_b6k_b9k_b12k_odf_path = os.path.join(base_save_path,'b3k_b6k_b9k_b12k_shore_fodf_sh.mat')
savemat(b3k_b6k_b9k_b12k_odf_path,mdict={'b3k_b6k_b9k_b12k_gqi_odf_sh':b3k_b6k_b9k_b12k_shore_odfs_sh})



