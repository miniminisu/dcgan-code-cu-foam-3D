import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN3D_D(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN3D_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),  # (1,64,4,2,1)in_channels输入图像通道数, out_channels输出图像通道数也就是滤波器的个数, kernel_size, stride=1,padding=0,
            nn.LeakyReLU(0.2, inplace=True),
        )
        i, csize, cndf = 3, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cndf))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(out_feat))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=3
            cndf = cndf * 2
            csize = csize / 2
        # 全连接层
        # state size. K x 4 x 4 x 4
        main.add_module(str(i),
                        nn.Conv3d(cndf, 1, 4, 1, 0, bias=False))
        main.add_module(str(i+1), nn.Sigmoid())
        
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)

class DCGAN3D_G(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0): # isize是输入图像的大小，ngf是生成器滤波器的数量
        super(DCGAN3D_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4  # ngf=64, cngf=32, tisize初始为4
        while tisize != isize:  # tisize从 4 到8 16 32 64 128  (5次，代表5层网络)   isize=128
            cngf = cngf * 2  # cngf从 32到64,128,256,512,1024
            tisize = tisize * 2  # # tisize从 4 到8 16 32 64 128
        # 这一层应该是全连接层，把z变成(1024,4x4)
        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=False),
            nn.BatchNorm3d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf   # csize是卷积核的大小(4x4x4), cngf已从32变为1024
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

class DCGAN3D_G_CPU(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN3D_G_CPU, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=True),
            nn.BatchNorm3d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=True))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)