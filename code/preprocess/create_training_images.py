import tifffile
import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='copper_foam_tif', required=False, help='path to image')  # berea.tiff图片路径
parser.add_argument('--name', default='copperfoam', required=False, help='name of dataset')  # brea材料名
parser.add_argument('--edgelength', type=int, default=256, help='input batch size')  # 图像维度
parser.add_argument('--stride', type=int, default=50, help='the height / width of the input image to network')  # 高/宽

parser.add_argument('--target_dir', default='copper_foam_256', required=False, help='path to store training images')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.target_dir):
    os.mkdir(opt.target_dir)

count = 0

edge_length = opt.edgelength  # image dimensions 图像维度
stride = opt.stride  # stride at which images are extracted 图像提取的步幅

N = edge_length
M = edge_length
O = edge_length

I_inc = stride
J_inc = stride
K_inc = stride

target_direc = str(opt.target_dir)
count = 0
# 创建训练数据，格式是hdf5
# 抽取数据的时候，我们以正视图的角度去看，每stride长度抽取一个小立方体(64x64x64)。
# root = 'copper_foam'
file_names = os.listdir(str(opt.data_path))
for file_name in file_names:
    # 读取.tiff文件 berea.tiff是400x400x400
    file_path = os.path.join(opt.data_path, file_name)
    img = tifffile.imread(str(file_path))
    for i in range(0, img.shape[0], I_inc):
        for j in range(0, img.shape[1], J_inc):
            for k in range(0, img.shape[2], K_inc):
                subset = img[i:i+N, j:j+N, k:k+O]  # [0:64,0:64,0:64]
                if subset.shape == (N, M, O):
                    f = h5py.File(target_direc+"/"+str(opt.name)+"_"+str(count)+".hdf5", "w")
                    f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
                    f.close()
                    count += 1
    print(file_name, '文件可以切割出：', count, '个')
