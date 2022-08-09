import torch
from torch import nn
from d2l import torch as d2l


def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim,1))
    v_b = torch.zeros(1)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum']*v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()


data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)


def train_momentum(lr, momentum, num_epochs = 2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim), {'lr':lr, 'momentum': momentum}, data_iter, feature_dim, num_epochs)

train_momentum(0.02, 0.5)
