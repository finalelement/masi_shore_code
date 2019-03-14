import numpy as np
import os
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.direction import peaks_from_model

from scipy.io import loadmat,savemat

def reconst_gqi_fodf_sh_coeffs(bval_path,bvec_path,data_path,data_var):

    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)
    data_path = os.path.normpath(data_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs)

    data = loadmat(data_path)
    data = data[data_var]

    gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=1.2)

    #gqfit = gqmodel.fit(dataslice, mask=mask)
    sphere = get_sphere('symmetric724')
    #ODF = gqfit.odf(sphere)
    #odf = gqmodel.fit(data).odf(sphere)

    gqpeaks = peaks_from_model(model=gqmodel,
                               data=data,
                               sphere=sphere,
                               relative_peak_threshold=.5,
                               min_separation_angle=25,
                               return_odf=True,
                               normalize_peaks=True)

    print('Debug here')
    fodfs = gqpeaks.odf
    return fodfs

def reconst_gqi_fodf_return_sh_coeffs(bval_path,bvec_path,data_path,data_var):

    bval_path = os.path.normpath(bval_path)
    bvec_path = os.path.normpath(bvec_path)
    data_path = os.path.normpath(data_path)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs)

    data = loadmat(data_path)
    data = data[data_var]

    gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=1.2)

    #gqfit = gqmodel.fit(dataslice, mask=mask)
    sphere = get_sphere('symmetric724')
    #ODF = gqfit.odf(sphere)
    #odf = gqmodel.fit(data).odf(sphere)

    gqpeaks = peaks_from_model(model=gqmodel,
                               data=data,
                               sphere=sphere,
                               relative_peak_threshold=.5,
                               min_separation_angle=25,
                               return_odf=True,
                               normalize_peaks=True)

    print('Debug here')
    fodfs_shm = gqpeaks.shm_coeff
    return fodfs_shm

def fiber_nav_sh_to_masi_sh_order_8th(sh_coeffs):

    new_order_8th_indices = [0, \
                             5, 4, 3, 2, 1, \
                             14, 13, 12 ,11, 10, 9, 8, 7, 6, \
                             27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, \
                             44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28]

    flip_sign_indices = [2, 4, \
                         7, 9, 11, 13, \
                         16, 18, 20, 22, 24, 26, \
                         29, 31, 33, 35, 37, 39, 41, 43]

    new_sh_coeffs = np.zeros((len(sh_coeffs),45))

    new_sh_coeffs = sh_coeffs[:, new_order_8th_indices]
    new_sh_coeffs[:, flip_sign_indices] = -1 * new_sh_coeffs[:, flip_sign_indices]

    return new_sh_coeffs









