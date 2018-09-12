# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import mnist_dataset, two_class_dataset
from model import Layer
import matplotlib.pyplot as plt
from optimizer import *
from loss import *


def convert_y(y, target_num):
    res = np.zeros(y.shape)
    res[y == target_num] = 1
    return res


def mnist():
    class_num = 2
    n_trains = 6000
    X_train, Y_train, X_test, Y_test = mnist_dataset(n_trains)
    Y_train = convert_y(Y_train, class_num)
    Y_test = convert_y(Y_test, class_num)
    # print(X_train.shape, Y_train.shape)
    model = Layer(X_train.shape[0], 1, activation='sigmoid') \
        .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam()) \
        .fit(X_train, Y_train, epoch=100, batch_size=int(n_trains / 10))
    plt.plot(range(len(model.losses)), model.losses, color='orange')
    plt.show()


def two_class():
    X_train, Y_train = two_class_dataset()
    model = Layer(X_train.shape[0], 1, activation='sigmoid') \
        .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam()) \
        .fit(X_train, Y_train, epoch=200, batch_size=8)
    plt.plot(range(len(model.losses)), model.losses, color='orange')
    plt.show()


if __name__ == '__main__':
    two_class()


'''

'''
