import torch
import numpy
from d2l import torch as d2l
import random

"""
pretest
"""
class HyperParameters:
    def save_hyperparameters(self, ignore = []):
        raise NotImplemented


"""
Create a SyntheticRegressionDate
"""


class DataModule(d2l.HyperParameters):
    def __init__(self, root = '../data', num_workers = 4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplemented

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloeader(self):
        return self.get_dataloader(train = False)


class SyntheticRegressionData(d2l.DataModle):
    def __init__(self, w, b, noise = 0.01, num_train = 1000, num_val = 1000, batch_size = 32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

if __name__ == '__main__':
    data = SyntheticRegressionData(w = torch.tensor([2, -3.4]), b = 4.2)
    print(("features:", data.X[0], '\nlabel:', data.y[0]))