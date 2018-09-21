# encoding=utf8
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_var(shape):
    initital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initital)


def bias_var(shape):
    initital = tf.constant(0.1, shape=[shape])
    return tf.Variable(initital)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class CnnLayer:

    def __init__(self, shape):
        self.w = weight_var(shape)
        self.b = bias_var(shape[-1])

    def forward(self, x):
        linear = conv2d(x, self.w) + self.b
        conv = tf.nn.relu(linear)
        pool = max_pool_2x2(conv)
        return pool


class FullLayer:

    def __init__(self, shape, activation=None):
        self.w = weight_var(shape)
        self.b = bias_var(shape[-1])
        self.activation = activation

    def forward(self, x):
        linear = tf.matmul(x, self.w) + self.b
        if self.activation is None:
            return linear
        return tf.nn.relu(linear)


def init_holder():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    return x, y, keep_prob


def create_graph(x, y, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
    a1 = CnnLayer([5, 5, 1, 32]).forward(x_image)
    # h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 64)
    a2 = CnnLayer([5, 5, 32, 64]).forward(a1)
    # h_pool2(batch, 7, 7, 64) -> h_fc1(1, 1024)
    flat_n = 7 * 7 * 64
    bb = FullLayer([flat_n, 1024], activation='relu').forward(tf.reshape(a2, [-1, flat_n]))
    a3 = tf.nn.dropout(bb, keep_prob)
    # 1024 -> 10
    y_predict = tf.nn.softmax(FullLayer([1024, 10]).forward(a3))

    cross_entropy = -tf.reduce_sum(y * tf.log(y_predict))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return train_step, accuracy, cross_entropy


def train(x, y, keep_prob, train_step, accuracy, cross_entropy):
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = mnist.train.next_batch(500)
        if i % 100 == 0:
            loss = cross_entropy.eval(session=sess,
                                      feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print("step %d, loss %g" % (i, loss))
            train_step.run(session=sess,
                           feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(session=sess,
                                             feed_dict={x: mnist.test.images, y: mnist.test.labels,
                                                        keep_prob: 1.0}))

if __name__ == '__main__':
    x, y, keep_prob = init_holder()
    train_step, accuracy, cross_entropy = create_graph(x, y, keep_prob)
    train(x, y, keep_prob, train_step, accuracy, cross_entropy)

'''

'''
