import numpy as np
import h5py
import tifffile
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu
from collections import Counter
import os
# 获取hdf5所有path
root_hdf5 = r'fake_images/hdf5'
root_tiff = r'fake_images/tiff'
root_postprocess_tiff = r'fake_images/postprocess_tiff'
file_paths = os.listdir(root_hdf5)
# 遍历所有path，读取图像
for path in file_paths:
    # 读取hdf5图像数据，并取出图像内容
    f = h5py.File(os.path.join(root_hdf5, path), 'r')
    my_array = f['data'][()]
    img = my_array[0, 0, :, :, :].astype(np.float32)  # 原始数据是5维的ndarray，但是这里的hdf5文件却是3维
    path = path.split('.hdf5')[0]
    path = path+'.tiff'
    tifffile.imsave(os.path.join(root_tiff, path), img)
    print(img.shape)

# 读取tiff图像,并做处理
file_paths = os.listdir(root_tiff)
for path in file_paths:
    im_in = tifffile.imread(os.path.join(root_tiff, path))
    # 中值滤波处理
    im_in = median_filter(im_in, size=(3, 3, 3))
    # 裁剪之外的噪声区域 cutaway outer noise area
    # im_in = im_in[40:240, 40:240, 40:240]
    # 归一化为0-1的小数
    im_in = im_in/255.
    # 阈值处理
    threshold_global_otsu = threshold_otsu(im_in)
    segmented_image = (im_in >= threshold_global_otsu).astype(np.int32)
    # 保存处理后的图像
    tifffile.imsave(os.path.join(root_postprocess_tiff, 'postprocess'+path), segmented_image.astype(np.int32))

# # 计算孔隙度
# segmented_image = tifffile.imread("postprocessed_example.tiff")
# porc = Counter(segmented_image.flatten())
# print(porc)
# porosity = porc[0]/float(porc[0]+porc[1])
# print("Porosity of the sample: ", porosity)