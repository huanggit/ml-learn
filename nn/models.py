# encoding=utf8
import numpy as np
from layer import Layer
from math import floor


class Model:

    good_enough_loss = 0.045

    def __init__(self):
        self.losses = list()

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


class LinearModel(Model):
    """docstring for ClassName"""

    def __init__(self, n_features, activation=None):
        Model.__init__(self)
        self.layer = Layer(n_features, 1, activation)

    def init_optimizers_for_layers(self, optimizer):
        self.optimizer_name = optimizer.__class__.__name__
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

    def __init__(self, n_nodes, activation=None):
        Model.__init__(self)
        self.layers = list()
        for i in range(len(n_nodes) - 1):
            self.layers.append(Layer(n_nodes[i], n_nodes[i + 1], activation))
        self.layers.append(Layer(n_nodes[-1], 1, 'sigmoid'))

    def init_optimizers_for_layers(self, optimizer):
        self.optimizer_name = optimizer.__class__.__name__
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
