import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter


class VModel(nn.Module):
    """Specifies the way the state value function is estimated, and the
    corresponding constraint.
    """

    def g_v(self, state):
        """Returns a list of unit vectors. Each vector is the normalized
        gradient of the value function in the parameter with same index in
        self.parameters().
        """

        self.zero_grad()
        val = self.forward(state)  # compute value function
        val.backward()
        return [self.normalize(param.grad) for param in self.parameters()]

    @staticmethod
    def normalize(vect):
        """L2-normalization"""
        return vect / torch.norm(vect, p=2)


class LinearBaird(VModel):
    def __init__(self):
        super(LinearBaird, self).__init__()
        self.M = self.init_m()

        self.nparams = 7
        theta = torch.rand(self.nparams)  # init randomly in [0,1]

        # self.theta = Variable(theta)
        self.register_parameter("theta", Parameter(theta))

    def forward(self, state, action=None):  # ignore action for Q methods
        return torch.dot(self.M[state, :], self.theta)

    @staticmethod
    def init_m():
        M = Variable(torch.zeros(6, 7))
        # Each line i : the parameters associated to state i
        M[:, 0] = 1
        for i in range(5):
            M[i, i + 1] = 2
        M[5, 0] = 2
        M[5, 6] = 1
        return M


if __name__ == "__main__":
    value = LinearBaird()
    optimizer = optim.SGD(value.parameters(), lr=0.1)
    
    print(value.M)
    print(value.theta)
    print(value.g_v(2))