# encoding=utf8
import numpy as np
from layer import Layer
from math import floor
import matplotlib.pyplot as plt
from activation import *


class Model:

    def __init__(self, good_enough_loss=0.045):
        self.losses = list()
        self.good_enough_loss = good_enough_loss

    def compile(self, loss_func, optimizer):
        self.loss_func = loss_func
        self.init_optimizers_for_layers(optimizer)
        return self

    def init_optimizers_for_layers(self, optimizer):
        raise Exception('abstract method should be implemented')

    def predict(self, x):
        raise Exception('abstract method should be implemented')

    def loss_value(self, x, y):
        loss = self.loss_func.loss_value(y, self.predict(x))
        self.losses.append(loss)
        return loss

    def fit(self, x, y, epoch, batch_size):
        steps_in_a_epoch = floor(x.shape[1] / batch_size)
        for step in range(epoch):
            if self.loss_value(x, y) < self.good_enough_loss:
                break
            for i in range(steps_in_a_epoch):
                start = i * batch_size
                end = start + batch_size
                x_batch, y_batch = x[:, start:end], y[:, start:end]
                self.step(x_batch, y_batch)
        return self

    def step(self, x_batch, y_batch):
        raise Exception('abstract method should be implemented')

    def classify_y(self, x):
        Yp = self.predict(x)
        Y_predict = np.zeros(Yp.shape)
        Y_predict[Yp > 0.5] = 1
        return Y_predict

    def plot_decision_boundary(self, x, y):
        x_min, x_max = x[0].min() - .5, x[0].max() + .5
        y_min, y_max = x[1].min() - .5, x[1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.classify_y(np.array([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(x[0].ravel(), x[1].ravel(), c=y.ravel(), cmap=plt.cm.Spectral)
        return self

    def plot_loss(self):
        plt.plot(range(len(self.losses)), self.losses, color='orange')
        return self

    def plot(self, x, y):
        fig = plt.figure(1, figsize=(9, 4))
        fig.suptitle('Model Info', fontsize=15)
        plt.subplot(1, 2, 1)
        self.plot_loss()
        plt.subplot(1, 2, 2)
        self.plot_decision_boundary(x, y)
        return self

    def show(self):
        plt.show()


class LinearModel(Model):
    """docstring for ClassName"""

    def __init__(self, n_features, activation=Tanh(), good_enough_loss=0.045):
        Model.__init__(self, good_enough_loss)
        self.layer = Layer(n_features, 1, activation)

    def init_optimizers_for_layers(self, optimizer):
        self.layer.compile(optimizer)

    def predict(self, x):
        A = self.layer.forward(x)
        return A

    def step(self, x_batch, y_batch):
        A = self.layer.forward(x_batch)
        dA = self.loss_func.derivative(y_batch, A)
        dA_prev, dW, db = self.layer.backward(dA)
        self.layer.update_params(dW, db)


class MultiLayerModel(Model):
    """docstring for ClassName"""

    def __init__(self, n_nodes,
                 activation=Tanh(),
                 initialization="he",
                 good_enough_loss=0.045):
        Model.__init__(self, good_enough_loss)
        self.layers = list()
        for i in range(len(n_nodes) - 2):
            self.layers.append(Layer(n_nodes[i], n_nodes[i + 1], activation, initialization))
        self.layers.append(Layer(n_nodes[-2], n_nodes[-1], Sigmoid(), initialization))

    def init_optimizers_for_layers(self, optimizer):
        for layer in self.layers:
            layer.compile(optimizer)

    def predict(self, x):
        A = x
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def step(self, x_batch, y_batch):
        A = x_batch
        for layer in self.layers:
            A = layer.forward(A)
        dA = self.loss_func.derivative(y_batch, A)
        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA)
            layer.update_params(dW, db)
