# encoding=utf8
import numpy as np


class BaseOptimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def init_shape(self, n_features, n_hidden):
        pass

    def update_w(self, delta):
        pass


class SGD(BaseOptimizer):
    """stochastic gradient descent"""

    def __init__(self, learning_rate=0.1):
        BaseOptimizer.__init__(self, learning_rate)

    def update_w(self, delta):
        return self.learning_rate * delta


class Momentum(BaseOptimizer):
    """Momentum"""

    def __init__(self, learning_rate=0.1, gamma=0.8):
        BaseOptimizer.__init__(self, learning_rate)
        self.gamma = gamma

    def init_shape(self, n_features, n_hidden):
        self.v = np.zeros((n_hidden, n_features + 1))

    def update_w(self, delta):
        self.v = self.gamma * self.v + self.learning_rate * delta
        return self.v


class AdaGrad(BaseOptimizer):
    """AdaGrad: learning_rate can be large at start, and it will decay."""

    def __init__(self, learning_rate=0.1, epsilon=1e-8):
        BaseOptimizer.__init__(self, learning_rate)
        self.epsilon = epsilon

    def init_shape(self, n_features, n_hidden):
        self.accumulate = np.zeros((n_hidden, n_features + 1))

    def update_w(self, delta):
        self.accumulate += delta ** 2
        return delta * self.learning_rate / np.sqrt(self.accumulate + self.epsilon)


class Adam(BaseOptimizer):
    """Adam"""

    def __init__(self, learning_rate=0.1, epsilon=1e-8, beta1=0.9, beta2=0.999):
        BaseOptimizer.__init__(self, learning_rate)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def init_shape(self, n_features, n_hidden):
        self.m = np.zeros((n_hidden, n_features + 1))
        self.v = np.zeros((n_hidden, n_features + 1))

    def update_w(self, delta):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        mb = self.m / (1 - self.beta1**self.t)
        vb = self.v / (1 - self.beta2**self.t)
        return self.learning_rate * mb / np.sqrt(vb + self.epsilon)


'''

'''
