import torch
from torch import nn as nn
from torch.nn import functional as F

class Double2dConv(nn.Module):
    def __init__(self, in_channels, out_channels, down = False):
        super(Double2dConv, self).__init__()
        self.down = down
        if in_channels!=out_channels:
            self.res = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res = nn.Identity()
        self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels//2),
        nn.Dropout2d(p=0.15),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(p=0.15),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        if self.down:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        res = self.res(x)
        out = self.double_conv(x)
        res_out = res + out
        if self.down:
            final_out = self.pooling(res_out)
            return final_out, res_out
        else:
            return res_out
class Up2dConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up2dConv, self).__init__()
        if in_channels!=out_channels:
            self.res = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res = nn.Identity()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(p=0.15),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(p=0.15),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),)
    def forward(self, x, x_encoder):
        x = self.up(x)
        x = torch.cat((x, x_encoder), dim=1)
        res = self.res(x)
        out = self.double_conv(x)
        res_out = res + out
        return res_out

class Backbone(nn.Module):
    def __init__(self, in_c):
        super(Backbone, self).__init__()
        self.in_c = in_c
        self.conv1 = Double2dConv(in_channels=self.in_c, out_channels=32, down = True)  
        self.conv2 = Double2dConv(in_channels=32, out_channels=64, down = True)  
        self.conv3 = Double2dConv(in_channels=64, out_channels=128, down = True) 
        self.conv4 = Double2dConv(in_channels=128, out_channels=256, down = True)  
        self.conv5 = Double2dConv(in_channels=256, out_channels=512, down = False)  
        self.deconv4 = Up2dConv(in_channels=512, out_channels=256)
        self.deconv3 = Up2dConv(in_channels=256, out_channels=128)
        self.deconv2 = Up2dConv(in_channels=128, out_channels=64)
        self.deconv1 = Up2dConv(in_channels=64, out_channels=32)
        self.cls = nn.Conv2d(32, 2, kernel_size=1)
    def forward(self, x):
        x1_down, x1 = self.conv1(x) 
        x2_down, x2 = self.conv2(x1_down) 
        x3_down, x3 = self.conv3(x2_down) 
        x4_down, x4 = self.conv4(x3_down) 
        x5 = self.conv5(x4_down)
        x4_up = self.deconv4(x5, x4) 
        x3_up = self.deconv3(x4_up, x3) 
        x2_up = self.deconv2(x3_up, x2)
        x1_up = self.deconv1(x2_up, x1)
        out = self.cls(x1_up)
        return out, x1_up











