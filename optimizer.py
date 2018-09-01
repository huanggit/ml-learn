# encoding=utf8
import numpy as np


class BaseOptimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class SGD(BaseOptimizer):
    """stochastic gradient descent"""

    def __init__(self, learning_rate=0.1):
        super(SGD, self).__init__(learning_rate)

    def update_w(self, delta):
        return self.learning_rate * delta


class Momentum(BaseOptimizer):
    """Momentum"""

    def __init__(self, learning_rate=0.1, gamma=0.8):
        super(Momentum, self).__init__(learning_rate)
        self.gamma = gamma
        self.v = np.array([0, 0])

    def update_w(self, delta):
        self.v = self.gamma * self.v + self.learning_rate * delta
        return self.v
