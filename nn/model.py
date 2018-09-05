# encoding=utf8
import numpy as np
from nn_util import add_ones


class LinearModel:
    '''y_predict = a * x + b'''

    good_enough_loss = 0.01

    def __init__(self, a=10.0, b=-20.0):
        self.losses = list()
        self.weight_init(a, b)

    def weight_init(self, a, b):
        self.w = np.array([a, b])

    def predict(self, x):
        return np.matmul(self.w, add_ones(x))

    def compile(self, loss_func, optimizer):
        self.loss_func = loss_func
        self.optimizer = optimizer
        return self

    def _loss(self, x, y):
        return np.round(self.loss_func.loss(y, self.predict(x)), 5)

    def _delta(self, x, y):
        return self.loss_func.delta(x, y, self.predict(x))

    def fit(self, dataset, epoch, batch_size=8):
        x, y = dataset.fetch()
        steps_in_a_epoch = int(len(x) / batch_size)
        for step in range(epoch):
            for i in range(steps_in_a_epoch):
                x_batch, y_batch = dataset.batch(batch_size)
                self.w -= self.optimizer.update_w(self._delta(x_batch, y_batch))
            loss = self._loss(x, y)
            self.losses.append(loss)
            if loss < self.good_enough_loss:
                break
        return self
