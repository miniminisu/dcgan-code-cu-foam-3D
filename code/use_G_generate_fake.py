from __future__ import print_function
import argparse
import os
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from dataset import HDF5Dataset
import dcgan
import utils


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
parser.add_argument('--netG', default='results_imageSize=64_batchSize=32_nz=500_ngf=64_ndf=8/netG_epoch_373.pth', help="path to netG (to continue training)")
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
    for i in range(1,6):
        # 1是64，5是128，13是256，22是400
        fixed_noise = torch.FloatTensor(1, nz, 13, 13, 13).normal_(0, 1)  # 生成图片用的噪声
        fixed_noise.cuda()  # GPU加速
        fake = netG(fixed_noise)  # 使用已经训练好的模型来生成图片
        utils.save_tiff(fake, out_dir + 'fake_size_'+str(fake.shape[2])+'_'+str(i)+'.tiff')
    print('5张重构生成完毕')