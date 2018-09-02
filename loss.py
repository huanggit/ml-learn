# encoding=utf8
import numpy as np


def loss_mse(y_true, y_predict):
    '''Mean Squared Error
    sum((y_predict - y_true)^2)/(2*size)
    '''
    size = len(y_true)
    return sum(np.square(np.round(y_predict - y_true, 4))) / (size * 2)
