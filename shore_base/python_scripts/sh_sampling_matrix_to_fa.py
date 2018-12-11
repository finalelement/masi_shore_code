import numpy as np
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.viz import window, actor

n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

# Enables/disables interactive visualization
interactive = True

ren = window.Renderer()
ren.SetBackground(1, 1, 1)

ren.add(actor.point(hsph_initial.vertices, window.colors.red, point_radius=0.05))
ren.add(actor.point(hsph_updated.vertices, window.colors.green, point_radius=0.05))

print('Saving illustration as initial_vs_updated.png')
window.record(ren, out_path='initial_vs_updated.png', size=(300, 300))
if interactive:
    window.show(ren)