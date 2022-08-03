import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # if the patthern is test, eps is very small
    if not torch.is_grad_enabled():
        X_hat = (X -moving_mean)/torch.sqrt(moving_var + eps)
    else:
        assert  len(X.shape) in (2, 4), 'wrong' # the feature of X will from linear or convolutional network
        if len(X.shape) == 2:
            # we need to comput on axis = 1, the column
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # we need to comput on axis = 1, the column
            mean = X.mean(dim=(0,2,3), keepdim = True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim = True)
        X_hat = (X - mean)/ torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features : c_o
    # num_dims: 2: fully connections, 4 cnn
    def __init__(self, num_feature, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_feature) # (n, c_o)
        else:
            shape = (1, num_feature, 1, 1) # (n, c_o, h, w)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_mean.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


class Resudal(nn.Module):
    def __init__(self,in_channels, num_channels, use_1x1conv = True, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1,stride=stride)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )

def ResNet_block(in_channels, num_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Resudal(in_channels, num_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Resudal(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*ResNet_block(64, 64, 2, first_block=True)) # this function is to avoid '/2' : stride = 2
b3 = nn.Sequential(*ResNet_block(64, 128, 2))
b4 = nn.Sequential(*ResNet_block(128, 256, 2))
b5 = nn.Sequential(*ResNet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(512, 10)
                    )

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)



