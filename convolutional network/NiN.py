import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,strides, padding),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
                         )

net = nn.Sequential(
        nin_block(1, 96, 11, 4,0),
        nn.MaxPool2d(3,stride=2),
        nin_block(96, 256, 5, 1, 2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, 3, 1, 1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384,10,3,1,1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
)

X = torch.randn(1, 1, 224, 224)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)