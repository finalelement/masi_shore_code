import numpy as np
import os
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.direction import peaks_from_model

fetch_taiwan_ntu_dsi()
img, gtab = read_taiwan_ntu_dsi()

print(gtab.bvals)
print('################ \n')

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

affine = img.affine

voxel_size = img.header.get_zooms()[:3]

gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=3)

dataslice = data[:, :, data.shape[2] // 2]

mask = dataslice[..., 0] > 50

gqfit = gqmodel.fit(dataslice, mask=mask)

sphere = get_sphere('symmetric724')

ODF = gqfit.odf(sphere)

print('ODF.shape (%d, %d, %d)' % ODF.shape)

gqpeaks = peaks_from_model(model=gqmodel,
                           data=dataslice,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=False,
                           normalize_peaks=True)

gqpeak_values = gqpeaks.peak_values

gqpeak_indices = gqpeaks.peak_indices

GFA = gqpeaks.gfa

print('GFA.shape (%d, %d)' % GFA.shape)

gqpeaks = peaks_from_model(model=gqmodel,
                           data=dataslice,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=True,
                           normalize_peaks=True)