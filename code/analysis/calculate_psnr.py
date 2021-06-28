import os
import tifffile
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import matplotlib.pyplot as plt

fake_root_path = 'psnr_data/fake_64'
real_root_path = 'psnr_data/real_64'
compared_root_path = 'compared/copperfoam_0.tiff'
fake_image_names = os.listdir(fake_root_path)
real_image_names = os.listdir(real_root_path)
compared_img = tifffile.imread(compared_root_path)
print('fake and compared')
fake_psnr_list = []
for fake_image_name in fake_image_names:
    fake_image_path = os.path.join(fake_root_path, fake_image_name)
    fake_img = tifffile.imread(fake_image_path)
    # mse = peak_signal_noise_ratio((fake_img / 255).astype(np.int), (compared_img / 255).astype(np.int))
    mse = peak_signal_noise_ratio(fake_img.astype(np.uint8), compared_img.astype(np.uint8))
    mse = round(mse, 3)
    fake_psnr_list.append(mse)
    # print(mse)

print('real and compared')

real_psnr_list = []
for real_image_name in real_image_names:
    real_image_path = os.path.join(real_root_path, real_image_name)
    real_img = tifffile.imread(real_image_path)
    # mse = peak_signal_noise_ratio((real_img / 255).astype(np.int), (compared_img / 255).astype(np.int))
    mse = peak_signal_noise_ratio(real_img.astype(np.uint8), compared_img.astype(np.uint8))
    mse = round(mse, 3)
    real_psnr_list.append(mse)
    # print(mse)

# PSNR的偏差
fake_img_mean = np.mean(fake_psnr_list)
fake_img_sigma = np.std(fake_psnr_list)
real_img_mean = np.mean(real_psnr_list)
real_img_sigma = np.std(real_psnr_list)
print('fake_img_mean:', fake_img_mean, 'fake_img_sigma:', fake_img_sigma)
print('real_img_mean:', real_img_mean, 'real_img_sigma:', real_img_sigma)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(4, 4))  # 设置画布的尺寸
plt.title('峰值信噪比', fontsize=20)  # 标题，并设定字号大小
labels = '生成图像', '真实图像'  # 图例
plt.boxplot([fake_psnr_list, real_psnr_list], labels=labels)  # grid=False：代表不显示背景中的网格线
plt.show()  # 显示图像
