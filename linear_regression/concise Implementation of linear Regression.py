import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l


"""
concise implementation of linear regression
"""


def load_array(data_arrays, batch_size, is_train = True):
    '''turn to be iteration'''

    dataset = data.TensorDataset(*data_arrays) # debagging the tuple
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    num_epochs = 3

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        L = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss{L:f}')

w = net[0].weight.data
b = net[0].bias.data
print(f'the loss about estimate w:{true_w - w.reshape(true_w.shape)}')
print(f'the loss about estimate b:{true_b - b}')
