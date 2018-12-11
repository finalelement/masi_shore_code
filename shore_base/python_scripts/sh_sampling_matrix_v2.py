import numpy as np
import math as m
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.sphere import Sphere, HemiSphere
from dipy.viz import window, actor
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import SphHarmModel
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import QballBaseModel
from dipy.reconst.shm import sf_to_sh
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import mean_diffusivity


'''
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az
'''

def cart2sphA(pts):
    return np.array([cart2sphere(x,y,z) for x,y,z in pts])

'''
def appendSpherical(xyz):
    np.hstack((xyz, cart2sphA(xyz)))
'''


# Load Gradient Directions and Visualize in theta and phi coordinates on a hemisphere
bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec'
bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval'

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

print("Debug Here")

# Calculate Spherical Coordinates
b3000_bvecs = bvecs[0:100,:]
b3000_bvals = bvals[0:100]
test = cart2sphA(b3000_bvecs)
theta = test[:,1]
phi = test[:,2]

# Gradient Table formation for the heck of it
gtab_b3000 = gradient_table(b3000_bvals, b3000_bvecs)

# Generate a Hemisphere figure plot
hsph_initial = HemiSphere(theta=theta, phi=phi)

# Enables/disables interactive visualization
interactive = True

ren = window.Renderer()
ren.SetBackground(1, 1, 1)

ren.add(actor.point(hsph_initial.vertices, window.colors.red, point_radius=0.05))

if interactive:
    window.show(ren)

print('The Points should be looking good visually !')

print('Lets form the much needed sphere')

sphere_grad = Sphere(xyz=np.vstack((hsph_initial.vertices, -hsph_initial.vertices)))

window.rm_all(ren)
ren.add(actor.point(sphere_grad.vertices, actor.colors.green, point_radius=0.05))
if interactive:
    window.show(ren)

# Lets make the Sampling Matrix, We will first create a SH object
sh_model = SphHarmModel(gtab_b3000)

# Trying to get Basis matrix using Qball method is a failed idea because there are checks on the fit matrix.
#q_sh_model = QballBaseModel(gtab_b3000,sh_order=8,smooth=0.006)
#q_sh_basis = q_sh_model.sampling_matrix(hsph_initial)

sh_basis = sh_model.sampling_matrix(hsph_initial)

print('Debug the Spherical Harmonic Basis Set')

# Lets try the SF to SH method
fake_data = np.ones((1,100))
fake_sh_model, fake_sh_basis = sf_to_sh(fake_data, hsph_initial, sh_order=8, basis_type='fibernav')

print('Debug Fake SH Model')

print('Recreate Fake SH Data')

pred_fake_data = np.dot(fake_sh_basis, fake_sh_model.T)
pred_fake_data = pred_fake_data.T
print('Predictions of Signal Made')

# Fit Tensor and Calculate FA and MD of the data
tenmodel = dti.TensorModel(gtab_b3000)
tenfit = tenmodel.fit(pred_fake_data)

FA = fractional_anisotropy(tenfit.evals)
MD = mean_diffusivity(tenfit.evals)

print('Fractional Anisotropy \n')
print(FA)

print('Mean Diffusivity \n')
print(MD)