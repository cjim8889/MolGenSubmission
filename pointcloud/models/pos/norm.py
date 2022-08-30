from turtle import forward
from survae.transforms.bijections import Bijection
from survae.transforms import ActNormBijection1d
import torch
from torch import nn
from einops import rearrange


class ActNormFlow(ActNormBijection1d):
    def __init__(self, num_features, data_dep_init=True, eps=0.000001):
        super(ActNormFlow, self).__init__(num_features, data_dep_init, eps)


    def forward(self, x, mask=None, logs=None):
        x = rearrange(x, 'b d c -> b c d')
        z, log_det = super(ActNormFlow, self).forward(x)

        z = rearrange(z, 'b d c -> b c d')
        return z, log_det
    
    def inverse(self, x, mask=None):

        x = rearrange(x, 'b d c -> b c d')

        return rearrange(super(ActNormFlow, self).inverse(x), 'b d c -> b c d'), 0. 
