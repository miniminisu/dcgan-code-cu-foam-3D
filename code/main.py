from __future__ import print_function
import argparse
import os
import random
# import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
from torch.autograd import Variable
from dataset import HDF5Dataset
# from hdf5_io import save_hdf5
import dcgan
import numpy as np
import utils

np.random.seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3D', help='3D')
parser.add_argument('--dataroot', default='preprocess\copper_foam_64', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')  # 批处理大小
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')  # 训练数据图像的大小
parser.add_argument('--nz', type=int, default=500, help='size of the latent z vector')  # z的维度
parser.add_argument('--ngf', type=int, default=64)  # number of generator filter  # 生成器滤波7         器的数量
parser.add_argument('--ndf', type=int, default=8)  # number of discriminator filter  # 鉴别器滤波器的数量
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')  # epoch的数量
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')  # 学习率大小
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')  # adam优化器的参数
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')


opt = parser.parse_args()
print(opt)

out_dir = './results'+'_imageSize='+str(opt.imageSize)+'_batchSize='+str(opt.batchSize)+'_nz='+str(opt.nz)+'_ngf='+str(opt.ngf)+'_ndf='+str(opt.ndf)+'/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

utils.save_prgs(opt, out_dir)
# 结果保存在哪里


try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.manualSeed = 43  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# 将图像转为-1或者1的数值
if opt.dataset in ['3D']:
    dataset = HDF5Dataset(opt.dataroot, input_transform=transforms.Compose([transforms.ToTensor()]))
assert dataset
# 加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1  # 图像的通道数，我的数据不是RGB三通道，只是单通道黑白图像

# 这个函数可以用于各种类型的层选择不同的初始化的方式。
# custom weights initialization called on netG and netD
def weights_init(m):
    # 获得m实例的类名
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 创建一个DCGAN3D_G的对象
netG = dcgan.DCGAN3D_G(opt.imageSize, nz, nc, ngf, ngpu)
netG.apply(weights_init)  # 初始化G的权重

# 判断生成器G的模型路径是否为空
# 如果路径不为空，那么就去加载这个路径下的模型参数
if opt.netG != '':
    print('通过加载模型来生成10张图像')
    netG.load_state_dict(torch.load(opt.netG))  # 加载已训练的模型
    netG.cuda()  # GPU加速
    for i in range(10):
        fixed_noise = torch.FloatTensor(1, nz, 13, 13, 13).normal_(0, 1)  # 生成图片用的噪声
        fixed_noise.cuda()  # GPU加速
        fake = netG(fixed_noise)  # 使用已经训练好的模型来生成图片
        utils.save_tiff(fake, out_dir + 'fake_generate_again'+str(i)+'.tiff')
    print('10张重构生成完毕')
# 打印生成器G的结构
print(netG)

# 创建一个DCGAN3D_D的对象
netD = dcgan.DCGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu)
netD.apply(weights_init)  # # 初始化D的权重
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)
# 交叉熵损失
criterion = nn.BCELoss()
# 创建z变量
# input是每次训练需要喂入的数据[128,1,64,64,64]
# noise是每次训练喂入的噪声[128,512,1,1,1],512是z的维度，z可以更改
# fixed_noise[1,512,7,7,7]
# fixed_noise_TI[1,512,1,1,1]
input, noise, fixed_noise, fixed_noise_TI = None, None, None, None
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1, 1)  # 训练用的噪声
fixed_noise = torch.FloatTensor(1, nz, 5, 5, 5).normal_(0, 1)  # 生成图片用的噪声
fixed_noise_TI = torch.FloatTensor(1, nz, 1, 1, 1).normal_(0, 1)  # 生成图片用的噪声

label = torch.FloatTensor(opt.batchSize)
real_label = 1
# real_label = np.random.uniform(0.7, 0.9)
fake_label = 0
# fake_label = np.random.uniform(0, 0.3)
# 让变量可以用GPU计算
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_noise_TI = fixed_noise_TI.cuda()

# 创建好变量
input = Variable(input)
label = Variable(label)
noise = Variable(noise)

fixed_noise = Variable(fixed_noise)
fixed_noise_TI = Variable(fixed_noise_TI)

# 创建优化器
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# 损失文件路径
loss_csv_path = out_dir + "training_curve.csv"
# 开始训练!!!
gen_iterations = 0
for epoch in range(opt.niter):
    # 迭代一批训练数据
    for i, data in enumerate(dataloader, 0):
        # 把loss写入csv文件

        f = open(out_dir + "training_curve.csv", "a")

        ############################
        # (1) 更新D网络: maximize log(D(x)) + log(1 - D(G(z))) 这个值一开始肯定非常的大，因为D(X)为1,D(G(z))为0。训练刚开始，左边一坨非常大，右边一坨也非常大
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        # input.data.resize_(real_cpu.size()).copy_(real_cpu)
        # label.data.resize_(batch_size).fill_(real_label)
        input.resize_(real_cpu.size()).copy_(real_cpu)  # 一批真实数据()(16,1,128,128)
        label.resize_(batch_size).fill_(real_label)  # 真实数据的标签设置为(16个0.9)
        
        output = netD(input)  # 这里相当于是D(x)
        errD_real = criterion(output.squeeze(), label)  # 计算交叉熵损失 log(D(z))
        errD_real.backward()  # 反向传播
        D_x = output.data.mean()  # 顺带求一下D(x)的均值为多少，鉴别器认为x为真的概率

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1, 1)
        # noise噪声z初始化为0,1之间的小数
        noise.resize_(batch_size, nz, 1, 1, 1)
        noise.data.normal_(0, 1)  # 噪声z为均值为0，标准差为1的正太分布数，但是为什么还是有大于1的数字?
        fake = netG(noise).detach()  # G(z)
        # 标签平滑处理
        label.data.fill_(fake_label)
        output = netD(fake)  # D(G(z))
        errD_fake = criterion(output.squeeze(), label)  # 交叉熵损失
        errD_fake.backward()  # 反向传播
        D_G_z1 = output.data.mean()  # 顺便求一下D(G(z))的均值
        errD = errD_real + errD_fake  # 公式两边相加
        optimizerD.step()  # 优化D
        
        ############################
        # (2) 更新G网络: maximize log(D(G(z)))
        ###########################
        g_iter = 1
        while g_iter != 0:
            netG.zero_grad()
            label.data.fill_(1.0)  # fake labels are real for generator cost
            noise.data.normal_(0, 1)  # 这个noise与上面的noise肯定不一样
            fake = netG(noise)
            output = netD(fake)
            errG = criterion(output.squeeze(), label)
            errG.backward()
            D_G_z2 = output.data.mean()  # # 顺便求一下D(G(z))的均值
            optimizerG.step()  # 优化G
            g_iter -= 1
        
        gen_iterations += 1

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 # errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                 errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2))  # D(G(z1)是先更新鉴别器后计算的值，D(G(z2))是后更新生成器计算的值)
        f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader),
                   # errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                   errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2))
        f.write('\n')
        f.close()
    # 每5个epoch生成中间结果和断点模型
    if epoch % 1 == 0:
        # 生成图片
        # fake_z = netG(noise)
        fake = netG(fixed_noise)
        # fake_batch = netG(batch_noise[0])
        fake_TI = netG(fixed_noise_TI)
        # for i in range(len(fake_batch)):
        utils.save_tiff(fake_TI, out_dir+'fake_TI_{0}.tiff'.format(epoch))
        utils.save_tiff(fake, out_dir + 'fake_{0}.tiff'.format(epoch))
        # save_hdf5(fake_TI.data, work_dir+'fake_batch_{0}.tiff'.format(epoch))
        # save_hdf5(fake_z.data, work_dir + 'fake_z_{0}.hdf5'.format(epoch))
        # save_hdf5(fake.data, work_dir+'fake_samples_{0}.hdf5'.format(epoch))
        # save_hdf5(fake_TI.data, work_dir+'fake_TI_{0}.hdf5'.format(epoch))
        # 保存模型断点
        torch.save(netG.state_dict(), out_dir + 'netG_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), out_dir + 'netD_epoch_%d.pth' % epoch)

# 损失曲线绘图
utils.save_learning_curve(out_dir)

