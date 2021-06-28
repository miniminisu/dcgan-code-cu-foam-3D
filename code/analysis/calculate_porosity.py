import os
import tifffile
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

fake_root_path = 'fake_64'
real_root_path = 'real_64'
image_names = os.listdir(fake_root_path)

fake_porosity_list = []
for image_name in image_names:
    image_path = os.path.join(fake_root_path, image_name)
    img = tifffile.imread(image_path)
    porosity = 1 - ps.metrics.porosity(img / 255)
    porosity = round(porosity, 3)
    print(porosity)
    fake_porosity_list.append(porosity)

print('----------')
image_names = os.listdir(real_root_path)
real_porosity_list = []
for image_name in image_names:
    image_path = os.path.join(real_root_path, image_name)
    img = tifffile.imread(image_path)
    porosity = 1 - ps.metrics.porosity(img / 255)
    porosity = round(porosity, 3)
    print(porosity)
    real_porosity_list.append(porosity)

# 输出孔隙率均值和方差
print('真实图像孔隙率均值：', np.mean(real_porosity_list))
print('孔隙率方差：', np.std(real_porosity_list))

print('生成图像孔隙率均值：', np.mean(fake_porosity_list))
print('孔隙率方差：', np.std(fake_porosity_list))
# 绘图

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(4, 4))  # 设置画布的尺寸
plt.title('孔隙率')  # 标题，并设定字号大小
labels = '重建图像', '真实图像'  # 图例
plt.boxplot([fake_porosity_list, real_porosity_list], labels=labels)  # grid=False：代表不显示背景中的网格线
# data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
plt.show()  # 显示图像
