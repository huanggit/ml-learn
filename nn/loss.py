# encoding=utf8
import numpy as np


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
        size = len(y_true)
        return -1 * sum(np.multiply(y_true, np.log(y_predict))
                        + np.multiply(1 - y_true, np.log(1 - y_predict))) / size

    def derivative(self, y_true, y_predict):
        return -1 * (np.divide(y_true, y_predict) - np.divide(1 - y_true, 1 - y_predict))
