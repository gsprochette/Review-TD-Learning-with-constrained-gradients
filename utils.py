import pickle
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


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
