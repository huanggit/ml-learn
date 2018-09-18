# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import *
from models import LinearModel, MultiLayerModel
import matplotlib.pyplot as plt
from optimizer import *
from loss import *


def two_class_none_linear():
    fig = plt.figure(1, figsize=(8, 8))
    X_train, Y_train = moons_dataset()
    for inx, optimizer in enumerate([SGD(0.8), Momentum(0.3), AdaGrad(0.3), Adam(0.1)]):
        model = MultiLayerModel([X_train.shape[0], 10, 5, 1], activation='tanh') \
            .compile(loss_func=CROSS_ENTROPY(), optimizer=optimizer) \
            .fit(X_train, Y_train, epoch=3000, batch_size=256)
        plt.subplot(4, 2, 1 + 2 * inx)
        model.plot_loss()
        plt.subplot(4, 2, 2 + 2 * inx)
        model.plot_decision_boundary(X_train, Y_train)
    plt.show()


if __name__ == '__main__':
    two_class_none_linear()


'''

'''
