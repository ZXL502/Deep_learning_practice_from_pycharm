import torch
from torch import nn
from d2l import torch as d2l

arch=((1,1,64), (1,64,128), (1,128,256), (2,256,512), (2,512,512), (2,512,512))


def VGG_blk(num_convs, in_channels, out_channels):
    blk = []
    for _ in range(num_convs):
        if _ == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

def VGG(arh):
    # convolutional network
    con_blks = []
    for (num_convs, in_channels, out_channels) in arh:
        con_blks.append(VGG_blk(num_convs, in_channels, out_channels))
    # dense layer
    return nn.Sequential(
        *con_blks,
        nn.Flatten(),
        nn.Linear(out_channels*3*3, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096),nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
                         )


net = VGG(arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

