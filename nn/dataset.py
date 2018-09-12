# encoding=utf8
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


def norm_array(arr):
    'make all array elements in range [0,1]'
    a_min, a_max = min(arr), max(arr)
    a_norm = a_max - a_min
    return (arr - a_min) / a_norm


def linear_dataset():
    x = [30	, 35, 37,	59,	70,	76,	88,	100]
    y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]
    x = norm_array(np.array(x))
    y = norm_array(np.array(y))
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return x, y


def two_class_dataset():
    x = [10, 15, 37,   28, 70, 76, 88, 100]
    y = np.array([1,  1,   1,   1,   0,   0,   0,   0])
    x = norm_array(np.array(x))
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return x, y


def mnist_dataset(train_num):
    mnist = fetch_mldata('MNIST original')
    X, Y = shuffle(mnist["data"], mnist["target"], random_state=0)
    X_train, Y_train = X[:train_num] / 255, Y[:train_num]
    X_test, Y_test = X[train_num:train_num + 100] / 255, Y[train_num:train_num + 100]
    return X_train.T, Y_train.reshape(1, -1), X_test.T, Y_test.reshape(1, -1)
