# encoding=utf8
import numpy as np
import matplotlib.pyplot as plot
from dataset import linear_dataset, model_sampling
from model import LinearModel
from optimizer import SGD, Momentum
from loss import loss_mse


def plot_data_and_model(data, model):
    # plot.rcParams['font.sans-serif'] = 'SimHei'
    # plot.rcParams['axes.unicode_minus'] = False
    plot.xlabel('x')
    plot.ylabel('y')
    # plot data
    plot.scatter(data.x, data.y)
    # plot model
    x_, y_ = model_sampling(model)
    plot.plot(x_, y_, color='green')
    plot.show()


def test_gradient_descent():
    model = LinearModel() \
        .compile(loss_func=loss_mse, optimizer=SGD()) \
        .fit(linear_dataset(), epoch=50)


def test_sgd():
    model = LinearModel() \
        .compile(loss_func=loss_mse, optimizer=SGD()) \
        .fit(linear_dataset(), epoch=50, batch_size=4)
    plot_data_and_model(linear_dataset(), model)


if __name__ == '__main__':
    model = LinearModel() \
        .compile(loss_func=loss_mse, optimizer=Momentum()) \
        .fit(linear_dataset(), epoch=50)
    plot_data_and_model(linear_dataset(), model)
