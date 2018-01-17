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

    def __init__(self):
        super(VModel, self).__init__()
        self.register_parameter("theta", None)

    def g_v(self, state):
        """Returns a list of unit vectors. Each vector is the normalized
        gradient of the value function in the parameter with same index in
        self.parameters().
        """

        self.zero_grad()
        val = self.forward(state)  # compute value function
        val.backward()
        return [param.grad / torch.norm(param.grad, p=2)
                for param in self.parameters()]


class LinearBaird(VModel):
    def __init__(self, theta=None):
        super(LinearBaird, self).__init__()
        self.M = self.init_m()

        self.nparams = 7
        if theta is None:
            theta = self.init_theta()
        else:
            assert theta.numel() == self.nparams
            assert isinstance(theta, torch.Tensor)
            theta = theta.clone()
        self.theta = Parameter(theta)

    def forward(self, state):  # ignore action for Q methods
        if isinstance(state, Variable):
            state = state.squeeze().data.numpy()
        return torch.dot(self.M[state, :], self.theta)

    def all_v(self):
        return torch.mv(self.M, self.theta)

    def init_theta(self):
        theta = torch.randn(self.nparams).abs()  # init chi-squared
        theta[-1] += 2  # predominent final value
        return theta

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # input is (x, y)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)  # output is (Q(-1), Q(+1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    value = LinearBaird()
    optimizer = optim.SGD(value.parameters(), lr=0.1)

    print(value.M)
    print(value.theta)
    print(value.g_v(2))
