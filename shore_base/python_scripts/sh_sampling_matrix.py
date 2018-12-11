import numpy as np
import math as m
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.sphere import Sphere, HemiSphere
from dipy.viz import window, actor

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az

def cart2sphA(pts):
    return np.array([cart2sph(x,y,z) for x,y,z in pts])

def appendSpherical(xyz):
    np.hstack((xyz, cart2sphA(xyz)))

# Load Gradient Directions and Visualize in theta and phi coordinates on a hemisphere
bvec_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec'
bval_path = r'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval'

bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

print("Debug Here")

# Calculate Spherical Coordinates
b3000_bvecs = bvecs[0:100,:]
test = cart2sphA(b3000_bvecs)
theta = test[:,1]
phi = test[:,2]

# Generate a Hemisphere figure plot
hsph_initial = HemiSphere(theta=theta, phi=phi)

# Enables/disables interactive visualization
interactive = True

ren = window.Renderer()
ren.SetBackground(1, 1, 1)

ren.add(actor.point(hsph_initial.vertices, window.colors.red, point_radius=0.05))

if interactive:
    window.show(ren)




