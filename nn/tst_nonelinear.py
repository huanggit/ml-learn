# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import *
from models import LinearModel, MultiLayerModel
import matplotlib.pyplot as plt
from optimizer import *
from loss import *
from activation import *


def moons_optimizers():
    fig = plt.figure(1, figsize=(8, 8))
    fig.suptitle('Optimizers', fontsize=15)
    X_train, Y_train = moons_dataset()
    for inx, optimizer in enumerate([SGD(0.8), Momentum(0.3), AdaGrad(0.3), Adam(0.1)]):
        model = MultiLayerModel([X_train.shape[0], 10, 5, 1]) \
            .compile(loss_func=CROSS_ENTROPY(), optimizer=optimizer) \
            .fit(X_train, Y_train, epoch=3000, batch_size=256)
        plt.subplot(4, 2, 1 + 2 * inx)
        plt.ylabel(optimizer.__class__.__name__)
        model.plot_loss()
        plt.subplot(4, 2, 2 + 2 * inx)
        model.plot_decision_boundary(X_train, Y_train)
    plt.show()


def moons_losses():
    fig = plt.figure(1, figsize=(8, 6))
    fig.suptitle('Losses', fontsize=15)
    X_train, Y_train = moons_dataset()
    for inx, loss_func in enumerate([MSE(), CROSS_ENTROPY()]):
        model = MultiLayerModel([X_train.shape[0], 10, 5, 1]) \
            .compile(loss_func=loss_func, optimizer=Adam(0.1)) \
            .fit(X_train, Y_train, epoch=800, batch_size=256)
        plt.subplot(2, 2, 1 + 2 * inx)
        plt.ylabel(loss_func.__class__.__name__)
        model.plot_loss()
        plt.subplot(2, 2, 2 + 2 * inx)
        model.plot_decision_boundary(X_train, Y_train)
    plt.show()


def moons_inits():
    fig = plt.figure(1, figsize=(8, 6))
    fig.suptitle('Losses', fontsize=15)
    X_train, Y_train = moons_dataset()
    for inx, initz in enumerate(['he', 'random']):
        model = MultiLayerModel([X_train.shape[0], 10, 5, 1], initialization=initz) \
            .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam(0.1)) \
            .fit(X_train, Y_train, epoch=400, batch_size=256)
        plt.subplot(2, 2, 1 + 2 * inx)
        plt.ylabel(initz)
        model.plot_loss()
        plt.subplot(2, 2, 2 + 2 * inx)
        model.plot_decision_boundary(X_train, Y_train)
    plt.show()


def moons_activations():
    fig = plt.figure(1, figsize=(6, 9))
    fig.suptitle('Activations', fontsize=15)
    X_train, Y_train = moons_dataset()
    for inx, activation in enumerate([NoneAct(), Sigmoid(),  Tanh(), Relu(), LeakyRelu()]):
        model = MultiLayerModel([X_train.shape[0], 10, 5, 1], activation=activation) \
            .compile(loss_func=CROSS_ENTROPY(), optimizer=Adam(0.1)) \
            .fit(X_train, Y_train, epoch=200, batch_size=256)
        plt.subplot(5, 2, 1 + 2 * inx)
        plt.ylabel(activation.__class__.__name__)
        model.plot_loss()
        plt.subplot(5, 2, 2 + 2 * inx)
        model.plot_decision_boundary(X_train, Y_train)
    plt.show()


if __name__ == '__main__':
    moons_activations()


'''

'''
