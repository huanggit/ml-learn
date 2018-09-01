# encoding=utf8
import numpy as np


def square(x):
    return x * x


def loss_mse(y_true, y_predict):
    '''Mean Squared Error
    sum((y_predict - y_true)^2)/(2*size)
    '''
    # if isinstance(y_true, np.float64):
    #     return square(rount(y_true - y_predict, 4)) / 2
    size = len(y_true)
    return sum(np.square(np.round(y_predict - y_true, 4))) / (size * 2)
