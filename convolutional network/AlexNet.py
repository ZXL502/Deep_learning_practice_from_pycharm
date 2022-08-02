import torch
from torch import nn
from d2l import torch as d2l

# model: imagenet

net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, padding= 1, stride=4), # 54
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride= 2), # 26
    nn.Conv2d(96, 256, kernel_size=5, padding=2), # 26, padding = 2 : p_h = 4, p_w = 4
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride= 2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride= 2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

X = torch.randn(1,3, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

if __name__ == '__main__':
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    
