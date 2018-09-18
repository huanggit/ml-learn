# encoding=utf8
import numpy as np
from numpy import random
from copy import deepcopy
from activation import *


class Layer:
    '''y_predict = a * x + b'''

    def __init__(self,
                 n_features,
                 n_hidden,
                 activation=NoneAct(),
                 keep_prob=None,
                 initialization="he"):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.weight_init(n_features, n_hidden, initialization)
        self.activation = activation
        self.keep_prob = keep_prob

    def weight_init(self, n_features, n_hidden, initialization="he"):
        if initialization == "he":
            self.w = random.RandomState(0).randn(n_hidden, n_features) / np.sqrt(n_features / 2)
        elif initialization == "random":
            self.w = random.RandomState(0).randn(n_hidden, n_features) * 0.1
        self.b = np.zeros((n_hidden, 1))

    def dropout_init(self, A):
        return (random.rand(A.shape[0], A.shape[1]) < self.keep_prob)

    def dropout(self, A):
        return (A * self.dropout_mat) / self.keep_prob

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
        A = self.activation.activate(Z)
        # A.shape == (n_hidden, n_batch_size)
        assert (A.shape == (self.w.shape[0], A_prev.shape[1]))
        self.Z = Z
        self.A_prev = A_prev
        if self.keep_prob is not None:
            self.dropout_mat = self.dropout_init(A)
            A = self.dropout(A)
        return A

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

    def backward(self, dA):
        if self.keep_prob is not None:
            dA = self.dropout(dA)
        dZ = self.activation.derivative(self.Z) * dA
        dA_prev, dW, db = self.linear_backward(dZ, self.A_prev)

        return dA_prev, dW, db

    def compile(self, optimizer):
        opz = deepcopy(optimizer)
        opz.init_shape(self.n_features, self.n_hidden)
        self.optimizer = opz

    def update_params(self, dW, db):
        update_w = self.optimizer.update_w(np.hstack([dW, db]))
        self.w -= update_w[:, :-1]
        self.b -= update_w[:, -1].reshape(-1, 1)
