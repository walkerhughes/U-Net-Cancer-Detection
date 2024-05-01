import torch 
import torch.nn as nn 

class ConvSteps(nn.Module): 
    """
    Helper class for sequential 2d-convolution -> batch-norm -> relu steps 
    """
    def __init__(self, in_channels, out_channels): 
        super(ConvSteps, self).__init__()
        self.conv_steps = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(), 
            nn.Conv2d(out_channels, out_channels, stride = 1, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
    def forward(self, x): 
        return self.conv_steps(x) 
    
class DownConv(nn.Module): 
    """
    Helper class for the down-convolutions on the left hand side of the U-Net architechture 
    """
    def __init__(self, in_channels, out_channels): 
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            ConvSteps(in_channels, out_channels)
        )
    def forward(self, x): 
        return self.down_conv(x) 

class UpConv(nn.Module): 
    """
    Helper class for the up-convolutions on the right hand side of the U-Net architechture 
    """
    def __init__(self, in_channels, out_channels): 
        super(UpConv, self).__init__()
        self.transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            stride = 2, 
            kernel_size = 2, 
            padding = 0
        )
        self.up_conv = ConvSteps(in_channels, out_channels)

    def forward(self, x, y):  
        return self.up_conv(torch.cat((self.transpose(x), y), dim = 1)) 
