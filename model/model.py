import torch 
import torch.nn as nn 
from model.convolutions import ConvSteps, DownConv, UpConv

class UNetCancerDetection(nn.Module):
    """
    U-Net Architecture composed of helper classes in ./convolutions.py
    """
    def __init__(self, dataset):
        super(UNetCancerDetection, self).__init__()
        self.__dict__.update(locals())

        # define as parameters the various steps with 
        # their respective dimensions 
        self.first = ConvSteps(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)

        # here are the down convolutions with 
        # concatenations and transposed convolutions 
        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        # last convolution 
        self.last = nn.Conv2d(64, 2, kernel_size = 1)

    def forward(self, input):
        
        # down convolutions 
        # each of these results in MaxPools and ConvSteps being called 
        conv0_out = self.first(input)
        conv1_out = self.down1(conv0_out)
        conv2_out = self.down2(conv1_out)
        conv3_out = self.down3(conv2_out)
        conv4_out = self.down4(conv3_out)

        # up convolutions / concatenations 
        # each results in ConvSteps and concatenations with a transpose 
        conv5_out = self.up1(conv4_out, conv3_out)
        conv6_out = self.up2(conv5_out, conv2_out)
        conv7_out = self.up3(conv6_out, conv1_out)
        conv8_out = self.up4(conv7_out, conv0_out)
        
        return self.last(conv8_out)