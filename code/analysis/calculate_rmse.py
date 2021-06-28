import os
import tifffile
from skimage.metrics import mean_squared_error, normalized_root_mse
import numpy as np
import matplotlib.pyplot as plt

fake_root_path = 'mse_data/fake_64'
real_root_path = 'mse_data/real_64'
compared_root_path = 'compared/copperfoam_0.tiff'
fake_image_names = os.listdir(fake_root_path)
real_image_names = os.listdir(real_root_path)
compared_img = tifffile.imread(compared_root_path)
print('fake and compared')
fake_mse_list = []
for fake_image_name in fake_image_names:
    fake_image_path = os.path.join(fake_root_path, fake_image_name)
    fake_img = tifffile.imread(fake_image_path)
    # mse = mean_squared_error((fake_img / 255).astype(np.int), (compared_img / 255).astype(np.int))
    mse = mean_squared_error(fake_img.astype(np.uint8), compared_img.astype(np.uint8))
    rmse = mse ** 0.5
    # mse = normalized_root_mse(fake_img.astype(np.uint8), compared_img.astype(np.uint8))
    # mse = mean_squared_error((fake_img / 255).astype(np.int), (compared_img / 255).astype(np.int))
    rmse = round(rmse, 3)
    fake_mse_list.append(rmse)
    # print(mse)

print('real and compared')

real_mse_list = []
for real_image_name in real_image_names:
    real_image_path = os.path.join(real_root_path, real_image_name)
    real_img = tifffile.imread(real_image_path)
    mse = mean_squared_error(real_img.astype(np.uint8), compared_img.astype(np.uint8))
    rmse = mse ** 0.5
    # mse = normalized_root_mse(real_img.astype(np.uint8), compared_img.astype(np.uint8))
    rmse = round(rmse, 3)
    real_mse_list.append(rmse)
    # print(mse)

# RMSE的偏差
fake_img_mean = np.mean(fake_mse_list)
fake_img_sigma = np.std(fake_mse_list)
real_img_mean = np.mean(real_mse_list)
real_img_sigma = np.std(real_mse_list)
print('fake_img_mean:', fake_img_mean, 'fake_img_sigma:', fake_img_sigma)
print('real_img_mean:', real_img_mean, 'real_img_sigma:', real_img_sigma)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(4, 4))  # 设置画布的尺寸
plt.title('均方根误差', fontsize=20)  # 标题，并设定字号大小
labels = '生成图像', '真实图像'  # 图例
plt.boxplot([fake_mse_list, real_mse_list], labels=labels)  # grid=False：代表不显示背景中的网格线
# data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
plt.show()  # 显示图像
# for fake_image_name, real_image_name in zip(fake_image_names, real_image_names):
#     fake_image_path = os.path.join(fake_root_path, fake_image_name)
#     real_image_path = os.path.join(real_root_path, real_image_name)
#
#     fake_img = tifffile.imread(fake_image_path)
#     real_img = tifffile.imread(real_image_path)
#     mse = mean_squared_error((fake_img / 255).astype(np.int), (real_img / 255).astype(np.int))
#     # mse = mse ** 0.5
#     mse = round(mse, 3)
#     print(mse)

# print('--------')
# for fake_image_name, real_image_name in zip(fake_image_names, real_image_names):
#     fake_image_path = os.path.join(fake_root_path, fake_image_name)
#     real_image_path = os.path.join(real_root_path, real_image_name)
#
#     fake_img = tifffile.imread(fake_image_path)
#     real_img = tifffile.imread(real_image_path)
#     nrmse = normalized_root_mse(fake_img/255, real_img/255)
#     nrmse = round(nrmse, 3)
#     print(nrmse)
# fake_list = []
# for image_name in image_names:
#     image_path = os.path.join(fake_root_path, image_name)
#     img = tifffile.imread(image_path)
#     fake_list.append(img)
#
# image_names = os.listdir(real_root_path)
# real_list = []
# for image_name in image_names:
#     image_path = os.path.join(real_root_path, image_name)
#     img = tifffile.imread(image_path)
#     real_list.append(img)
#
#
# mse = compare_mse(fake_list, real_list)
# print(mse)
