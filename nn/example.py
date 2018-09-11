# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import linear_dataset
from model import Layer
from optimizer import *
from loss import *


def model_sampling(model, start=0, end=1.1, interval=0.1):
    'to plot the model fit line, we need to sample some points in model'
    x_ = np.arange(start, end, interval)
    y_, Z, A_prev = model.forward(x_.reshape(1, -1))
    return x_, np.squeeze(y_)


def init_plot():
    # plot.rcParams['font.sans-serif'] = 'SimHei'
    # plot.rcParams['axes.unicode_minus'] = False
    fig = plot.figure(1, figsize=(14, 7))
    fig.suptitle('Optimizers', fontsize=15)


def plot_model(data, model, inx):
    def plot_data_and_model():
        plot.subplot(2, 4, inx + 1)
        plot.scatter(data.x, data.y)
        x_, y_ = model_sampling(model)
        plot.plot(x_, y_, color='green')
        plot.title(model.optimizer.__class__.__name__)

    def plot_losses():
        plot.subplot(2, 4, inx + 5)
        plot.plot(range(len(model.losses)), model.losses, color='orange')
    plot_data_and_model()
    plot_losses()


if __name__ == '__main__':
    init_plot()
    dataset = linear_dataset()
    for inx, optimizer in enumerate([SGD()]):
        # for inx, optimizer in enumerate([SGD(), Momentum(), AdaGrad(), Adam()]):
        model = Layer(1, 1) \
            .compile(loss_func=MSE(), optimizer=optimizer) \
            .fit(dataset, epoch=100)
        plot_model(dataset, model, inx)
    plot.show()


'''

'''
