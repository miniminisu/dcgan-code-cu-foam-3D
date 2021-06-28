import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import tifffile
plt.rcParams['font.sans-serif'] = ['SimHei']
img_path = 'porosity_xyz_data/The original image.tiff'
im = tifffile.imread(img_path)/255
im = im.astype(np.bool)
voxel_size = 1
x_profile = 1-ps.metrics.porosity_profile(im, 0)
y_profile = 1-ps.metrics.porosity_profile(im, 1)
z_profile = 1-ps.metrics.porosity_profile(im, 2)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(np.linspace(0, im.shape[0]*voxel_size, im.shape[0]), x_profile, 'b-', label='yz-平面', alpha=0.5)
ax.plot(np.linspace(0, im.shape[1]*voxel_size, im.shape[1]), y_profile, 'r-', label='xz-平面', alpha=0.5)
ax.plot(np.linspace(0, im.shape[2]*voxel_size, im.shape[2]), z_profile, 'g-', label='xy-平面', alpha=0.5)
ax.set_ylim([0, 1])
ax.set_ylabel('切片的孔隙度')
ax.set_xlabel('切片的位置')
ax.legend(loc='lower right');
plt.show()
