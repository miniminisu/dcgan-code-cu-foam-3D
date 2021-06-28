import numpy as np
# import h5py
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def save_tiff(tensor, path):
    tensor = tensor.cpu()
    img = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()  # (1,1,x,x,x)
    # 中值滤波处理 img[0,0,:,:,:]代表先切片
    # img = median_filter(img[0, 0, :, :, :], size=(3, 3, 3))
    # 归一化为0-1的小数
    # img = img / 255.
    # 阈值处理
    threshold_global_otsu = threshold_otsu(img)
    segmented_image = (img >= threshold_global_otsu).astype(np.uint8)

    segmented_image = median_filter(segmented_image[0, 0, :, :, :], size=(5, 5, 5))
    # img = ndarr['data'][()]
    # img = img[0, 0, :, :, :].astype(np.float32)  # 原始数据是5维的ndarray，但是这里的hdf5文件却是3维
    # path = path+'.tiff'
    tifffile.imsave(path, segmented_image*255)

def save_prgs(args, file_path):
    argsDict = args.__dict__
    with open(file_path+'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

def save_learning_curve(path):
    losses = []
    i = 0
    with open(path+"training_curve.csv", "r") as f:
        for i, row in enumerate(f):
            try:
                # print(row)
                lossD = float(row.split(" ")[2])
                lossG = float(row.split(" ")[4])
                errD = float(row.split(" ")[6])
                errG = float(row.split(" ")[8])
                losses.append((lossD, lossG, errD, errG))
            except IndexError:
                pass

    losses = np.array(losses)  # list里的每个元素都为1个tuple，而这个tuple里有4个元素，分别是lossD,loosG,errD,errG)，因此需要转换为(12153,4)的np数组
    print(losses.shape)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    n = 51
    w = 3
    labelsize = 24
    ticksize = 20
    legendsize = 20

    num_x = range(i)
    num_y = i

    ax.semilogy(num_x, medfilt(losses[0:num_y, 1], n), color="black", linewidth=w, label=r"$Generator$")
    ax.semilogy(num_x, medfilt(losses[0:num_y, 0], n), color="red", linewidth=w, label=r"$Discriminator$")

    ax.set_xlabel(r"$Generator \ Iterations$", fontsize=labelsize)
    ax.set_ylabel(r"$Discriminator/Generator \ Loss$", fontsize=labelsize)

    ax.legend(fontsize=legendsize, loc='upper right')
    plt.yscale('linear')  # 让y轴刻度为正常数值
    fig.savefig(path+"berea_training_curve.png", bbox_extra_artists=None, bbox_inches='tight', dpi=300)
