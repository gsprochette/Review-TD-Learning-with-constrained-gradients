"""Implements different Learning Rates classes"""

class LR:
    """Abstract learning rate class"""

    def __init__(self):
        self.val = 1.
        self.count = 0

    def update(self):
        pass

    def __mul__(self, other):
        return self.val * other

    def __rmul__(self, other):
        return self.val * other


class HarmonicLR(LR):
    """Learning rate $1 / (n+1)$"""

    def __init__(self, tau):
        super(HarmonicLR, self).__init__()
        self.tau = tau

    def update(self):
        self.count += 1
        self.val = 1. / (1 + float(self.count) / self.tau)


class RiemannLR(LR):
    """Learning rate $1 / (n+1)^\alpha$"""

    def __init__(self, alpha, tau):
        super(RiemannLR, self).__init__()
        self.alpha = alpha
        self.tau = tau

    def update(self):
        self.count += 1
        self.val = 1. / ((1 + self.count / self.tau) ** self.alpha)
