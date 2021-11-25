import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.tools import Identity

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return Identity()
    else:
        raise ValueError('Not supported activation:', name)