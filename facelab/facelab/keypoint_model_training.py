import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Keypoint_Class(nn.Module):
    def __init__(
        self,
        output_ch=3,
        img_ch=1,
        channels=[32,64,128,128,200],
        kernel=3,
        shape=(256, 256),
        n_upsample=2,
    ):
        super().__init__()
        self.n_upsample = n_upsample
        self.image_shape = shape
        self.kernel=kernel
        # self.bodyparts = labels_id
        self.bodyparts=['paw','spout']
        # self.device = device
        self.Conv = nn.Sequential()
        self.Conv.add_module("conv0",convblock(ch_in=img_ch, ch_out=channels[0], kernel_sz=self.kernel, block=0),)
        for k in range(1, len(channels)):
            # print("we are in conv channels[k - 1]",k,channels[k - 1])
            self.Conv.add_module(f"conv{k}",convblock(ch_in=channels[k - 1], ch_out=channels[k], kernel_sz=self.kernel, block=k),)

        self.Up_conv = nn.Sequential()
        for k in range(self.n_upsample):
            # print("we are in up,channels[-1 - k],channels[-2 - k]",channels[-1 - k],channels[-2 - k])
            self.Up_conv.add_module(f"upconv{k}",convblock(ch_in=channels[-1 - k] + channels[-2 - k],ch_out=channels[-2 - k],kernel_sz=self.kernel,),)

        self.Conv2_1x1 = nn.Sequential()
        for j in range(3):
            # print("we are in Conv2_1x1,channels[-2 - k]",channels[-2 - k])
            self.Conv2_1x1.add_module(f"conv{j}",nn.Conv2d(channels[-2 - k], output_ch, kernel_size=1, padding=0),)

    def forward(self, x, normalize=False, smooth_confidence=False, verbose=False):
        # encoding path
        xout = []
        # print("x in beginning ", x.shape)
        x = self.Conv[0](x)
        xout.append(x)
        for k in range(1, len(self.Conv)):
            x = F.max_pool2d(x, kernel_size=self.kernel, stride=2, padding=1)
            x = self.Conv[k](x)
            # print("x in pooling ",x.shape)
            xout.append(x)

        for k in range(len(self.Up_conv)):
            # print("        ")
            # print("before xup is", x.shape)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            # print("xup is second", x.shape)
            # print("cat shape",torch.cat((x, xout[-2 - k]), axis=1).shape)
            x = self.Up_conv[k](torch.cat((x, xout[-2 - k]), axis=1))
            # print("what is k in upsample cat",k)
            # print("xup is third cat", x.shape)

        locx = self.Conv2_1x1[1](x)
        locy = self.Conv2_1x1[2](x)
        hm   = self.Conv2_1x1[0](x)
        hm = F.relu(hm)


        return hm, locx, locy




class convblock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_sz, block=-1):
        super().__init__()
        self.conv = nn.Sequential()
        self.block = block
        if self.block != 0:
            self.conv.add_module("conv_0", batchconv(ch_in, ch_out, kernel_sz))
        else:
            self.conv.add_module("conv_0", batchconv0(ch_in, ch_out, kernel_sz))
        self.conv.add_module("conv_1", batchconv(ch_out, ch_out, kernel_sz))

    def forward(self, x):
        x = self.conv[1](self.conv[0](x))
        return x


def batchconv0(ch_in, ch_out, kernel_sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.Conv2d(ch_in, ch_out, kernel_sz, padding=kernel_sz // 2, bias=False),
    )


def batchconv(ch_in, ch_out, sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_in, ch_out, sz, padding=sz // 2, bias=False),
    )
