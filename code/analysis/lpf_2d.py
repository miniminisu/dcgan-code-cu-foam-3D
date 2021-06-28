import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
import os
import imageio

# im = ps.generators.blobs([400, 400], blobiness=[1, 2], porosity=0.6)
data_root_path = 'lpf_data_dcgan'
image_names = os.listdir(data_root_path)
im = imageio.imread(os.path.join(data_root_path, image_names[0]))
im = np.array(im).astype(np.bool)
# ps.imshow(im);

paths_x = ps.filters.distance_transform_lin(im, mode='forward', axis=0)
lpf_x = ps.metrics.linear_density(paths_x, bins=range(1, 100, 10))

paths_y = ps.filters.distance_transform_lin(im, mode='forward', axis=1)
lpf_y = ps.metrics.linear_density(paths_y, bins=range(1, 100, 10))
fig, ax = plt.subplots(figsize=(5, 6))
ax.bar(label='y axis', x=lpf_y.L, height=lpf_y.cdf, width=lpf_y.bin_widths, color='b', edgecolor='k', alpha=0.5)
ax.bar(label='x axis', x=lpf_x.L, height=lpf_x.cdf, width=lpf_x.bin_widths, color='r', edgecolor='k', alpha=0.5)

ax.set_xlabel('Path length [pixels]')
ax.set_ylabel('Fraction of pixels within stated distance to solid')

plt.legend(fontsize=15, loc='upper right')
plt.show()
