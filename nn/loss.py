# encoding=utf8
import numpy as np


def nzero(arr):
    return np.maximum(arr, 0.000001)


class LOSS_FUNC:

    def loss_value(self, y_true, y_predict):
        pass

    def derivative(self, y_true, y_predict):
        pass


class MSE(LOSS_FUNC):

    def loss_value(self, y_true, y_predict):
        '''Mean Squared Error
        sum((y_predict - y_true)^2)/(2*size)
        '''
        size = len(y_true)
        cost = np.sum(np.square(np.round(y_predict - y_true, 4))) / (size * 2)
        return np.squeeze(cost)

    def derivative(self, y_true, y_predict):
        return (y_predict - y_true)


class CROSS_ENTROPY(LOSS_FUNC):

    def loss_value(self, y_true, y_predict):
        size = y_true.shape[1]
        a = np.multiply(y_true, np.log(nzero(y_predict)))
        b = np.multiply(1 - y_true, np.log(nzero(1 - y_predict)))
        cost = -1 * np.sum(a + b) / size
        return np.squeeze(cost)

    def derivative(self, y_true, y_predict):
        a = np.divide(1 - y_true, nzero(1 - y_predict))
        b = np.divide(y_true, nzero(y_predict))
        return a - b
