import torch
from d2l import torch as d2l
from torch import nn

'''
init_weight
'''
num_inputs, num_outputs, num_hiddens = 784, 10, 256
w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)* 0.01) #randn 0-1 正太随机tensor
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)* 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1, b1, w2, b2]

'''
active function
'''


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


'''
model
'''


def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X@w1 + b1)
    return (H@w2+ b2)


'''
loss function
'''

loss = nn.CrossEntropyLoss(reduction='none')

'''
training
'''

if __name__ == '__main__':
    batch_size, num_epochs, lr = 256, 10, 0.1
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,updater)
