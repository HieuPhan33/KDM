import torch
from torch import nn
import torch.nn.functional as F

class GlobalAtt(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GlobalAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Downblock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, stride, padding):
        super(Downblock, self).__init__()

        self.dw = nn.Conv2d(
            channels,
            channels,
            groups=channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        # self.dw = nn.AvgPool2d(
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding
        # )
        self.bn = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels, channels // 8, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels // 8, out_channels, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.mlp(x)
        return x

class PyramidPooling(nn.Module):
    def __init__(self, region_size, channels, out_channels):
        super(PyramidPooling, self).__init__()

        # self.dw = nn.Conv2d(
        #     channels,
        #     channels,
        #     groups=channels,
        #     stride=stride,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     bias=False,
        # )
        self.dw = nn.AdaptiveAvgPool2d((region_size,region_size))
        self.bn = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels, channels // 8, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels // 8, out_channels=out_channels, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.mlp(x)
        return x

class SAModule(nn.Module):
    def __init__(self, channels, rates=(2,4,8)):
        super(SAModule, self).__init__()
        self.downops = nn.ModuleList()
        self.ga = GlobalAtt(channels,reduction=16)
        for r in rates:
            #downop = Downblock(channels, stride=r,kernel_size=r+1,padding=1)
            downop = PyramidPooling(region_size=r, channels=channels)
            self.downops.append(downop)

    def forward(self, x):
        # Down, up, sigmoid
        b,c,h,w = x.size()
        outs = self.ga(x)
        outs = F.interpolate(outs,(h,w))
        for i in range(len(self.downops)):
            o = self.downops[i](x)
            o = F.interpolate(o,(h,w))
            outs += o
        att = torch.sigmoid(outs)
        out = x * att
        return out


class SemanticGuidingSpectralAttention(nn.Module):
    def __init__(self,enc_channels, dec_channels, rates=(2,4,8)):
        super(SemanticGuidingSpectralAttention, self).__init__()
        self.downops = nn.ModuleList()
        self.conv_down = nn.Sequential(
            nn.Conv2d(dec_channels,enc_channels,kernel_size=1),
            nn.ReLU()
        )
        for r in rates:
            #downop = Downblock(enc_channels,enc_channels, stride=r, kernel_size=r + 1, padding=1)
            downop = PyramidPooling(region_size=r, channels=enc_channels,out_channels=enc_channels)
            self.downops.append(downop)

    def forward(self, x_dec, x_enc):
        b, c, h, w = x_enc.size()
        x_dec = self.conv_down(x_dec)
        outs = self.downops[0](x_dec)
        outs = F.interpolate(outs, (h, w))
        for i in range(1,len(self.downops)):
            o = self.downops[i](x_dec)
            o = F.interpolate(o, (h, w))
            outs += o
        att = F.sigmoid(outs)
        return att * x_enc