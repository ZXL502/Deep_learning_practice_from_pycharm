import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
import random

"""
pretest, wrong
"""


def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:

    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented

class DataModule():
    def __init__(self, root = '../data', num_workers = 4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplemented

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloeader(self):
        return self.get_dataloader(train = False)


"""
Create a SyntheticRegressionDate
"""


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) +b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))


"""
load the dataset
"""


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # [0, 1, 2......]
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]



'''
initialize the parameters from model 
'''
w = torch.normal(0, 0.01, size = (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


'''
Define model
'''
def linreg(X, W, b):
    ''' linear regression'''
    return torch.matmul(X, w) +b


'''
Loss function
'''

def squared_loss(y_hat, y):
    '''averaged loss'''
    return (y_hat - y.reshape(y_hat.shape))** 2 / 2

'''
optimizer
'''
def sgd(params, lr, batch_size):
    with torch.no_grad(): # needn't to compute and store the gradient for saving the resource
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() #graduent was turned to be zero



if __name__ == '__main__':
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X,"\n", y)
        break

    """
    training
    """

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward() # l.shape = (batch_size, 1)
            sgd([w,b], lr, batch_size)
        with torch.no_grad():
            train_1 = loss(net(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss{float(train_1.mean()):f}')

print(f'the loss about estimate w:{true_w - w.reshape(true_w.shape)}')
print(f'the loss about estimate b:{true_b - b}')

