import time
import porespy as ps

ps.visualization.set_mpl_style()
import os
import imageio
import numpy as np
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt

data_root_path = 'chords_data_dcgan'
image_names = os.listdir(data_root_path)
im = imageio.imread(os.path.join(data_root_path, image_names[0]))
im = np.array(im).astype(np.bool)
# im = ps.generators.blobs(shape=[400, 400], blobiness=[2, 1])
# fig, ax = plt.subplots()
# ax.imshow(im);
crds_x = ps.filters.apply_chords(im=im, spacing=1, axis=0)
crds_y = ps.filters.apply_chords(im=im, spacing=1, axis=1)
# fig, ax = plt.subplots(1, 2, figsize=[8, 4])
# ax[0].imshow(crds_x);
# ax[1].imshow(crds_y);

data_x = ps.metrics.chord_length_distribution(crds_x, bins=20)
data_y = ps.metrics.chord_length_distribution(crds_y, bins=20)

print(data_x._fields)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(label='y axis', x=data_y.L, height=data_y.cdf, width=data_y.bin_widths, color='b', edgecolor='k', alpha=0.5);
ax.bar(label='x axis', x=data_x.L, height=data_x.cdf, width=data_x.bin_widths, color='r', edgecolor='k', alpha=0.5);
ax.set_xlabel("Chord length")
ax.set_ylabel("Frequency")
plt.legend(fontsize=15, loc='upper right')
plt.show()
