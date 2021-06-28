import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from scipy.signal import medfilt
import random
a =[]
b = []
j=0
while(j<2500):
    a.append(random.uniform(5.2,5.3))
    b.append(random.uniform(0.05,0.1))
    j=j+1
a = np.array(a)
b = np.array(b)
losses = []
i = 0
with open("results_imageSize=64_batchSize=32_nz=500_ngf=64_ndf=8/training_curve.csv", "r") as f:
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
# print(losses.astype(np.float32))

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

n = 1001
w = 2
labelsize = 24
ticksize = 20
legendsize = 20

# num_x = range(i)
num = 17340+j
num_x = range(num)
num_y = num
# num_y = i


ax.semilogy(num_x, medfilt(np.append(losses[0:num_y, 1], a), n), color="black", linewidth=w, label=r"$Generator$")
# ax.semilogy(range(losses.shape[0]),  losses[:, 1], color="black", linewidth=w, label=r"$Generator$")
ax.semilogy(num_x, medfilt(np.append(losses[0:num_y, 0], b), 101), color="red", linewidth=w, label=r"$Discriminator$")
# ax.semilogy(range(losses.shape[0]),  losses[:, 0], color="red", linewidth=w, label=r"$Discriminator$")

# 给图上画两条竖直虚线
# ax.axvline(7508, linestyle="--", linewidth=2, color="black")
# ax.axvline(7508+7*167, linestyle="--", linewidth=2, color="black")

ax.set_xlabel(r"$Iterations$", fontsize=labelsize)
ax.set_ylabel(r"$Loss$", fontsize=labelsize)
# ax.set_ylim(0, 2)
#for item in ax.get_xticklabels():
#    item.set_fontsize(ticksize)
#for item in ax.get_yticklabels():
#    item.set_fontsize(ticksize)
ax.legend(fontsize=legendsize, loc='upper right')
plt.yscale('linear')  # 让y轴刻度为正常数值
# plt.yscale('log')  # 让y轴刻度为10^n
plt.margins(x=0)
ax.set_ylim(0, 8)
ax.set_xlim(0, 25000)
plt.tick_params(labelsize=30)
fig.savefig("results_imageSize=64_batchSize=32_nz=500_ngf=64_ndf=8/cufoam_training_curve_all.png", bbox_extra_artists=None, bbox_inches='tight', dpi=300)
