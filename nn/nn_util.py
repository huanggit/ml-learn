# encoding=utf8
import numpy as np


def batch_data(x, y, batch_size):
    shuffle_index = np.random.permutation(batch_size)
    return x[:shuffle_index], y[shuffle_index]


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _tanh(z):
    '''
    g(z)  = (e^z - e^-z) / (e^z + e^-z)
    g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
    '''
    pass


def _relu(z):
    return max(0, z)


def _leaky_relu(z):
    return max(0.01 * z, z)


def activate(activation, Z):
    if activation is None:
        return Z
    if activation == 'sigmoid':
        return _sigmoid(Z)
    if activation == 'relu':
        return _relu(Z)
    else:
        raise Exception('invalid activation')


def _sigmoid_derivative(Z):
    s = _sigmoid(Z)
    dZ = s * (1 - s)
    return dZ


def _relu_derivative(Z):
    dZ = np.zeros(Z.shape)
    dZ[Z > 0] = 1
    return dZ


def activate_derivative(activation, Z):
    if activation is None:
        return 1
    if activation == "relu":
        return _relu_derivative(Z)
    if activation == "sigmoid":
        return _sigmoid_derivative(Z)
