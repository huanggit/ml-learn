# encoding=utf8
import numpy as np
from nn_util import add_ones


class LOSS_FUNC:

    def loss(self, y_true, y_predict):
        pass

    def delta(self, y_true, y_predict, x):
        pass


class MSE(LOSS_FUNC):

    def loss(self, y_true, y_predict):
        '''Mean Squared Error
        sum((y_predict - y_true)^2)/(2*size)
        '''
        size = len(y_true)
        return sum(np.square(np.round(y_predict - y_true, 4))) / (size * 2)

    def delta(self, x, y_true, y_predict):
        return np.sum((y_predict - y_true) * add_ones(x), axis=1)
