import numpy as np
import os
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.direction import peaks_from_model
from dipy.reconst.shore import ShoreModel
from scipy.io import loadmat,savemat
from dipy.data import default_sphere

def reconstruct_shore_fodf(bval_path, bvec_path, data_path, data_var, zeta_val=700.0):

    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)
    data_path = os.path.normpath(data_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals,bvecs)

    data = loadmat(data_path)
    data = data[data_var]

    shore_model = ShoreModel(gtab,radial_order=6,zeta=zeta_val)

    shore_peaks = peaks_from_model(shore_model, data, default_sphere,
                                   relative_peak_threshold=.1,
                                   min_separation_angle=45)

    print('Debug Here')

    return shore_peaks.shm_coeff


