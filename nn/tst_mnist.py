# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import mnist_dataset, two_class_dataset
from models import LinearModel, MultiLayerModel
import matplotlib.pyplot as plt
from optimizer import *
from loss import *


def convert_y(y, target_num):
    res = np.zeros(y.shape)
    res[y == target_num] = 1
    return res


def plot_loss(model):
    plt.plot(range(len(model.losses)), model.losses, color='orange')
    plt.show()


def val_test(model, X_test, Y_test):
    Yp = model.predict(X_test)
    Y_predict = np.zeros(Yp.shape)
    Y_predict[Yp > 0.5] = 1
    metric = np.squeeze(Y_test == Y_predict)
    print('accuracy: {} / {}'.format(sum(metric), len(metric)))
    print('Y_test:\n{}\nY_predict:\n{}\nmetric:\n{}'.format(Y_test, Y_predict, metric))


def mnist():
    class_num = 4
    n_trains = 6000
    X_train, Y_train, X_test, Y_test = mnist_dataset(n_trains)
    Y_train = convert_y(Y_train, class_num)
    Y_test = convert_y(Y_test, class_num)

    model = MultiLayerModel([X_train.shape[0], 5])\
        .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam())\
        .fit(X_train, Y_train, epoch=100, batch_size=600)

    val_test(model, X_test, Y_test)
    # plot_loss(model)


def two_class_linear():
    X_train, Y_train = two_class_dataset()
    model = LinearModel(X_train.shape[0], activation='sigmoid') \
        .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam()) \
        .fit(X_train, Y_train, epoch=200, batch_size=8)
    plot_loss(model)


def two_class_two_layers():
    X_train, Y_train = two_class_dataset()
    model = MultiLayerModel([X_train.shape[0], 5]) \
        .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam()) \
        .fit(X_train, Y_train, epoch=100, batch_size=8)
    plot_loss(model)


if __name__ == '__main__':
    mnist()


'''

'''
