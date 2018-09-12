# encoding=utf8
import numpy as np
from layer import Layer
from copy import deepcopy
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
        pass

    def loss_value(self, x, y):
        pass

    def fit(self, x, y, epoch, batch_size):
        pass


class LinearModel(Model):
    """docstring for ClassName"""

    def __init__(self, n_features, activation=None):
        Model.__init__(self)
        self.n_features = n_features
        self.layer = Layer(n_features, 1, activation)

    def init_optimizers_for_layers(self, optz):
        optimizer = deepcopy(optz)
        optimizer.init_shape(self.n_features, 1)
        self.layer.compile(optimizer)

    def loss_value(self, x, y):
        A, Z, A_prev = self.layer.forward(x)
        loss = self.loss_func.loss_value(y, A)
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
                A, Z, A_prev = self.layer.forward(x_batch.copy())
                dA = self.loss_func.derivative(y_batch.copy(), A)
                dA_prev, dW, db = self.layer.backward(dA, Z, A_prev)
                self.layer.update_params(dW, db)
        return self
