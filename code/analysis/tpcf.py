import os
import tifffile

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio
from scipy import interpolate

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 材料图像根路径
    data_root_path = 'tpcf_data'
    image_names = os.listdir(data_root_path)
    # image_names = image_names[3], image_names[0], image_names[2], image_names[1]
    # 颜色列表
    color_list = []
    for color, color_hex in colors.cnames.items():
        color_list.append(color_hex)
    # 顏色下标
    i = 9
    # 新建一个名叫 Figure1的画图窗口
    plt.figure(1)
    #  遍历读取到的所有图片
    for name in image_names:
        image_path = os.path.join(data_root_path, name)
        print(image_path)
        image = tifffile.imread(image_path).astype(np.bool)

        # 计算两点相关函数值，两点之间的距离间隔取5，如果为1，那么计算量太大
        value = ps.metrics.two_point_correlation_bf(image, spacing=5)
        # 当距离为0的时候，两点相关函数值为孔隙度，这里的1-porosity才是我的图像的孔隙度
        value[1][0] = 1 - ps.metrics.porosity(image)
        # plt.plot(*value, 'b-o', label=name, color=color_list[i], marker=".")
        x_sample = value[0]  # 原始横坐标
        print('x_sample:', x_sample)
        # x坐标区间
        # np.linspace返回一个等差数列，返回190个x坐标
        xnew = np.linspace(x_sample.min(), x_sample.max(), 12)
        # print('xnew:', xnew)
        # y区间
        # 插值算法：数据量比较少，但是想用平滑曲线给描绘出来，这样就需要手动制造多余的数据
        func = interpolate.interp1d(x_sample, value[1], kind='cubic')  # 对value[1]两点相关函数进行插值
        ynew = func(xnew)  # 190个y数据
        # print('ynew:', len(ynew))
        # 开始绘图
        plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        if name == 'The original image.tiff':
            plt.plot(xnew, ynew, label='真实图像', color='#000000', marker="o")
        else:
            plt.plot(xnew, ynew, label='重建图像', color=color_list[i], marker="^")
        # x和y轴的最大和最小值
        plt.xlim(0, 60)
        plt.ylim(0, 1)
        i = i + 1
    # 添加x和y轴的标签
    plt.tick_params(labelsize=20)
    plt.xlabel("r(distance)", fontsize=15)
    plt.ylabel(r'$S_2$(r)', fontsize=15)
    plt.legend(fontsize=15, loc='upper right')
    plt.show()
