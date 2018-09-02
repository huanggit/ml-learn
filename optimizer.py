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


class AdaGrad(BaseOptimizer):
    """AdaGrad: learning_rate can be large at start, and it will decay."""

    def __init__(self, learning_rate=5, epsilon=1e-8):
        super(AdaGrad, self).__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulate = np.array([0.0, 0.0])

    def update_w(self, delta):
        self.accumulate += delta ** 2
        return delta * self.learning_rate / np.sqrt(self.accumulate + self.epsilon)


class Adam(BaseOptimizer):
    """Adam"""

    def __init__(self, learning_rate=5, epsilon=1e-8, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__(learning_rate)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.array([0.0, 0.0])
        self.v = np.array([0.0, 0.0])
        self.t = 0

    def update_w(self, delta):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        mb = self.m / (1 - self.beta1**self.t)
        vb = self.v / (1 - self.beta2**self.t)
        print('{}'.format(self.learning_rate * mb / np.sqrt(vb + self.epsilon) / delta))
        return self.learning_rate * mb / np.sqrt(vb + self.epsilon)


'''

'''
