from dipy.reconst.shore import ShoreModel
from dipy.viz import window, actor
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere

fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, 10:40]

print('data.shape (%d, %d, %d, %d)' % data.shape)

radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(gtab, radial_order=radial_order,
                 zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

asmfit = asm.fit(data_small)

sphere = get_sphere('symmetric724')

sh_coeffs = asmfit.odf_sh()
odf = asmfit.odf(sphere)
print('odf.shape (%d, %d, %d)' % odf.shape)

# Enables/disables interactive visualization
interactive = True

ren = window.Renderer()
sfu = actor.odf_slicer(odf[:, None, :], sphere=sphere, colormap='plasma', scale=0.5)
sfu.RotateX(-90)
sfu.display(y=0)
ren.add(sfu)
window.record(ren, out_path='odfs.png', size=(600, 600))
if interactive:
    window.show(ren)
