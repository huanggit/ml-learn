# encoding=utf8
import numpy as np
from sklearn.datasets import fetch_mldata
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


def linear_dataset():
    x = [30	, 35, 37,	59,	70,	76,	88,	100]
    y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]
    return Dataset(x, y)


def mnist_dataset():
    mnist = fetch_mldata('MNIST original')
    X, Y = mnist["data"], mnist["target"]
    train_num = 60000
    shuffle_index = random.permutation(train_num)
    X_train, Y_train = X[shuffle_index] / 255, Y[shuffle_index]
    X_test, Y_test = X[:train_num] / 255, Y[:train_num]
    return X_train, Y_train, X_test, Y_test
