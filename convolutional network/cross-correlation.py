import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, k):
    # the output is (N_h - K_h + 1) x (N_w -K_w +1)
    h, w = k.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j+w] * k).sum()
        return Y


# Convolutional Layers the weights and bias are fixed
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x,self.weight) + self.bias
'''

# the learning a Kernel
# contruct a two-dimensional convolutional, output = 1
conv2d1 = nn.LazyConv2d(1, kernel_size=(1, 2), bias= False)
# input = 1, output = 1
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias= False)
X = torch.ones((6, 8))
X[:, 2:6] = 0
k = torch.tensor([[1.0, -1.0]])
Y = corr2d(X,k)
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    l.sum().backward()
    conv2d.weight.data[:] -= lr*conv2d.weight.grad
    if(i + 1) % 2 == 0:
        print(f'epoch{i+1}, loss{l.sum():.3f}')
'''

# padding
# output_size = (N_h - K_h + p_h + 1) x (N_w - K_w + p_w + 1)
# p_h = k_h -1 p_w = k_w - 1 for giving the same hight and weight between input and output
# stride: [(N_h - K_h + p_h + s_h)/ s_h] x [(N_w - K_w + p_w + s_w)/s_w]
# ---> [(N_h + s_h - 1)/ s_h] x [(N_w + s_w - 1)/s_w]
# ---> (N_h/s_h) X (N_w/s_w)
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding= 1,)

# pooling
# output_size = (N_h - p_h + 1) x (N_w - p_w  + 1) p : pooling kernel
# output_size = (N_h - p_h + pa_h + s_h)/s_h x (N_w - p_w + pa_w + s_w)/s_w p : pooling kernel  s :stride pa: pad
# c_i = c_o

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()

    return Y
