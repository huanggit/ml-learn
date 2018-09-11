# encoding=utf8
import numpy as np
from numpy import random
from nn_util import *


class Layer:
    '''y_predict = a * x + b'''

    good_enough_loss = 0.01

    def __init__(self, n_features, n_hidden, activation=None):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.losses = list()
        self.weight_init(n_features, n_hidden)
        self.activation = activation

    def weight_init(self, n_features, n_hidden):
        self.w = random.randn(n_hidden, n_features) * 0.01
        self.b = np.zeros((n_hidden, 1))

    def linear_forward(self, A_prev):
        Z = np.dot(self.w, A_prev) + self.b
        return Z

    def forward(self, A_prev):
        '''
        input.shape == (n_features, n_batch_size)
        A.shape == Z.shape == (n_hidden, n_batch_size)
        '''
        assert (A_prev.shape[0] == self.n_features)
        Z = self.linear_forward(A_prev)
        A = activate(self.activation, Z)
        # A.shape == (n_hidden, n_batch_size)
        assert (A.shape == (self.w.shape[0], A_prev.shape[1]))
        return A, Z, A_prev

    def linear_backward(self, dZ, A_prev):
        '''
        Input da[l], Caches
        dZ[l] = dA[l] * g'[l](Z[l])
        dW[l] = (dZ[l]A[l-1].T) / m
        db[l] = sum(dZ[l])/m                
        dA[l-1] = w[l].T * dZ[l]           
        Output dA[l-1], dW[l], db[l]
        '''
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.w.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == self.w.shape)
        assert (db.shape == self.b.shape)
        return dA_prev, dW, db

    def backward(self, dA, Z, A_prev):
        dZ = activate_derivative(self.activation, Z) * dA
        dA_prev, dW, db = self.linear_backward(dZ, A_prev)
        return dA_prev, dW, db

    def compile(self, loss_func, optimizer):
        self.loss_func = loss_func
        self.optimizer = optimizer
        return self

    def update_params(self, dW, db):
        update_w = self.optimizer.update_w(np.vstack([dW, db]))
        self.w -= update_w[:-1]
        self.b -= update_w[-1].reshape(-1, 1)

    def loss_value(self, x, y):
        A, Z, A_prev = self.forward(x)
        return self.loss_func.loss_value(y, A)

    def fit(self, dataset, epoch, batch_size=8):
        x, y = dataset.fetch()
        x = x.reshape(self.n_features, -1)
        y = y.reshape(self.n_hidden, -1)
        steps_in_a_epoch = int(x.shape[1] / batch_size)
        for step in range(epoch):
            for i in range(steps_in_a_epoch):
                x_batch, y_batch = x, y
                A, Z, A_prev = self.forward(x_batch)
                dA = self.loss_func.derivative(y_batch, A)
                dA_prev, dW, db = self.backward(dA, Z, A_prev)
                self.update_params(dW, db)
            loss = self.loss_value(x, y)
            self.losses.append(loss)
            if loss < self.good_enough_loss:
                break
        return self
