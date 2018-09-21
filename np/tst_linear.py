# encoding=utf8
import numpy as np
import matplotlib.pyplot as plt
from dataset import linear_dataset
from models import LinearModel
from optimizer import *
from loss import *
from activation import *


class SimpleModel:

    def __init__(self):
        self.w = [0, 0]

    def predict(self, x):
        return self.w[0] * x + self.w[1]


def model_sampling(model, start=0, end=1.1, interval=0.1):
    'to plot the model fit line, we need to sample some points in model'
    x_ = np.arange(start, end, interval)
    y_ = model.predict(x_.reshape(1, -1))
    return x_, np.squeeze(y_)


def init_plot():
    # plt.rcParams['font.sans-serif'] = 'SimHei'
    # plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(1, figsize=(14, 7))
    fig.suptitle('Optimizers', fontsize=15)


COLOR = 0xDDDDDD


def plot_model(model):
    global COLOR
    x_, y_ = model_sampling(model, 0, 4)
    plt.plot(x_, y_, color='#{0:x}'.format(COLOR))
    COLOR -= 0x080808


def one_point_init():
    model = SimpleModel()
    x = np.array([3])
    y = np.array([3])
    plt.scatter(x, y)
    plot_model(model)
    y_ = model.predict(x)
    print('y = {}\ny` = {}\ny`-y = {}'.format(y, y_, y_ - y))
    print('either a +1 or b + 3 will do.')
    print('if we wanna increase both a and b:')
    print('a += 3*rate')
    print('b += 1*rate')
    rate = 0.1
    for i in range(5):
        model.w += rate * np.array([3, 1])
        print('a:{},b:{}'.format(model.w[0], model.w[1]))
        plot_model(model)
    plt.show()


def two_points_derivative():
    model = SimpleModel()
    x = np.array([3, 2])
    y = np.array([3, 2])
    plt.scatter(x, y)
    plot_model(model)
    rate = 0.02
    for i in range(12):
        y_diff = model.predict(x) - y
        model.w -= rate * np.sum(y_diff * np.array([x, np.ones(len(x))]), axis=1)
        plot_model(model)
    plt.show()


def four_points_derivative():
    model = SimpleModel()
    x = np.array([3, 2, 3.7, 1])
    y = np.array([2.6, 2.1, 3.5, 1.2])
    plt.scatter(x, y)
    plot_model(model)
    rate = 0.01
    for i in range(10):
        y_diff = model.predict(x) - y
        model.w -= rate * np.sum(y_diff * np.array([x, np.ones(len(x))]), axis=1)
        plot_model(model)
    plt.show()


def plot_m(x, y, model, inx):
    def plot_data_and_model():
        plt.subplot(2, 4, inx + 1)
        plt.scatter(np.squeeze(x), np.squeeze(y))
        x_, y_ = model_sampling(model)
        plt.plot(x_, y_, color='green')
        plt.title(model.layer.optimizer.__class__.__name__)

    def plot_losses():
        plt.subplot(2, 4, inx + 5)
        plt.plot(range(len(model.losses)), model.losses, color='orange')
    plot_data_and_model()
    plot_losses()


def optimizers():
    init_plot()
    x, y = linear_dataset()
    for inx, optimizer in enumerate([SGD(), Momentum(), AdaGrad(), Adam()]):
        model = LinearModel(n_features=1) \
            .compile(loss_func=MSE(), optimizer=optimizer) \
            .fit(x, y, epoch=100, batch_size=8)
        plot_m(x, y, model, inx)
    plt.show()


if __name__ == '__main__':
    four_points_derivative()

'''

'''
