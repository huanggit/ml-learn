# encoding=utf8
import numpy as np
import random


def norm_array(arr):
    'make all array elements in range [0,1]'
    a_min, a_max = min(arr), max(arr)
    a_norm = a_max - a_min
    return (arr - a_min) / a_norm


def shuffle(array, seed):
    random.seed(seed)
    random.shuffle(array)


class Dataset:

    def __init__(self, x, y):
        self.x = norm_array(np.array(x))
        self.y = norm_array(np.array(y))

    def fetch(self):
        return self.x, self.y

    def batch(self, batch_size):
        seed = random.random()
        shuffle(self.x, seed)
        shuffle(self.y, seed)
        return self.x[:batch_size], self.y[:batch_size]


def model_sampling(model):
    'to plot the model fit line, we need to sample some points in model'
    x_ = np.arange(0, 1.1, 0.1)
    y_ = model.predict(x_)
    return x_, y_


def linear_dataset():
    x = [30	, 35, 37,	59,	70,	76,	88,	100]
    y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]
    return Dataset(x, y)
