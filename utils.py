import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


def hook_(grad, message, print_grad=False):
    print(message)
    if print_grad:
        print(grad)


def hook(message, print_grad=False):
    return lambda grad: hook_(grad, message, print_grad)
