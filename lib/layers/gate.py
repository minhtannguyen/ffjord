import math
import torch
import torch.nn as nn

# Feedforward-Gate (FFGate-I)
class FeedforwardGateI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel, out_channel=10):
        super(FeedforwardGateI, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(out_channel, out_channel, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.avg_layer = nn.AdaptiveAvgPool2d((1,1))
        self.linear_layer = nn.Conv2d(in_channels=out_channel, out_channels=1,
                                      kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        tol = 1e-5 + 9e-5 * self.sigmoid(x)
        
        return tol
    
# Feedforward-Gate (FFGate-II)
class FeedforwardGateII(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel, out_channel=10):
        super(FeedforwardGateII, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(out_channel, out_channel, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.avg_layer = nn.AdaptiveAvgPool2d((1,1))
        self.linear_layer = nn.Conv2d(in_channels=out_channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        x = self.sigmoid(x)
    
        mean_tol = 1e-5 + 9e-5 * x[:,0]
        std_tol = (5.0/3.0) * 1e-5 * x[:,1]
        
        return mean_tol, std_tol
    
# FFGate-III
class FeedforwardGateIII(nn.Module):
    """ use single conv (stride=2) layer only"""
    def __init__(self, in_channel, out_channel=10):
        super(FeedforwardGateII, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = conv3x3(in_channel, out_channel, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.avg_layer = nn.AdaptiveAvgPool2d((1,1))
        self.linear_layer = nn.Conv2d(in_channels=out_channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        tol = self.linear_layer(x).squeeze()

        return tol

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)