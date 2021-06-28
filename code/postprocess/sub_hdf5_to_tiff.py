import numpy as np
import h5py
import tifffile
import os
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu
from collections import Counter

root_hdf5 = '../preprocess/copper_foam_256'
root_tiff = 'sub_images_tiff_256'

files_name = os.listdir(root_hdf5)

for file_name in files_name:
    file_path = os.path.join(root_hdf5, file_name)
    f = h5py.File(file_path, 'r')
    my_array = f['data'][()]
    img = my_array[:, :, :].astype(np.float32)
    file_name = file_name.split('.')[0]+".tiff"
    # print(name)
    file_path = os.path.join(root_tiff, file_name)
    tifffile.imsave(file_path, img)
