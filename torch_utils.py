import torch
import torch.nn as nn


def init_weights_xavier(weights):
    # init weights with xavier uniform initi
    return nn.init.xavier_uniform_(weights)


def init_weights_zeros(weights):
    # init weights with zeros value
    return nn.init.zeros_(weights)


def init_weights_normal(weights, mean=0.0, std=1.0):
    # init weights with normal distribution
    return nn.init.normal_(weights, mean=mean, std=std)


def euclidean_loss(input1, input2, multiplier=100, use_l1_l2=False, eps=0.01):
    loss = multiplier * input1 - multiplier * input2
    if use_l1_l2:
        loss = torch.mean(eps * loss ** 2 + torch.abs(loss))
    else:
        loss = torch.mean(loss ** 2)
    return loss


def zero_padding(shape, conv1d = False):
    if(conv1d):
        return nn.ConstantPad1d(shape,0)
    else:
        return nn.ZeroPad2d(shape)


#########!TODO custom layer norm function
