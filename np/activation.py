# encoding=utf8
import numpy as np


class ACTIVATION:

    def activate(self, x):
        pass

    def derivative(self, y):
        pass


class NoneAct(ACTIVATION):

    def activate(self, x):
        return x

    def derivative(self, y):
        return 1


class Sigmoid(ACTIVATION):

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, y):
        s = self.activate(y)
        dZ = s * (1 - s)
        return dZ


class Tanh(ACTIVATION):

    def activate(self, x):
        # ez = np.exp(x)
        # e_z = np.exp(-x)
        # return np.divide(ez - e_z, ez + e_z)
        return np.tanh(x)

    def derivative(self, y):
        return 1 - np.tanh(y) ** 2


class Relu(ACTIVATION):

    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, y):
        dZ = np.zeros(y.shape)
        dZ[y > 0] = 1
        return dZ


class LeakyRelu(ACTIVATION):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activate(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, y):
        dZ = np.full(y.shape, self.alpha)
        dZ[y > 0] = 1
        return dZ
