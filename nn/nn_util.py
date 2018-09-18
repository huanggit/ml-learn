# encoding=utf8
import numpy as np


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _tanh(z):
    ez = np.exp(z)
    e_z = np.exp(-z)
    return np.divide(ez - e_z, ez + e_z)


def _relu(z):
    return np.maximum(0, z)


def _leaky_relu(z):
    return np.maximum(0.01 * z, z)


def activate(activation, Z):
    if activation is None:
        return Z
    if activation == 'sigmoid':
        return _sigmoid(Z)
    if activation == 'relu':
        return _relu(Z)
    if activation == 'leaky_relu':
        return _leaky_relu(Z)
    if activation == 'tanh':
        return np.tanh(Z)
    else:
        raise Exception('invalid activation')


def _sigmoid_derivative(Z):
    s = _sigmoid(Z)
    dZ = s * (1 - s)
    return dZ


def _relu_derivative(Z):
    dZ = np.full(Z.shape, 0.01)
    dZ[Z > 0] = 1
    return dZ


def _leaky_relu_derivative(Z):
    dZ = np.zeros(Z.shape)
    dZ[Z > 0] = 1
    return dZ


def _tanh_derivative(Z):
    return 1 - np.tanh(Z) ** 2


def activate_derivative(activation, Z):
    if activation is None:
        return 1
    if activation == "sigmoid":
        return _sigmoid_derivative(Z)
    if activation == "tanh":
        return _tanh_derivative(Z)
    if activation == "relu":
        return _relu_derivative(Z)
    if activation == 'leaky_relu':
        return _relu_derivative(Z)
    else:
        raise Exception('invalid activation')
