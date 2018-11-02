# encoding=utf8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time


def weight_var(shape):
    initital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initital)


def bias_var(shape):
    initital = tf.constant(0.1, shape=[shape])
    return tf.Variable(initital)


class CnnLayer:

    def __init__(self, shape):
        self.w = weight_var(shape)
        self.b = bias_var(shape[-1])

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def forward(self, x):
        linear = CnnLayer.conv2d(x, self.w) + self.b
        conv = tf.nn.relu(linear)
        pool = CnnLayer.max_pool_2x2(conv)
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


class CNN():

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.create_graph()

    def create_graph(self):
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
        a1 = CnnLayer([5, 5, 1, 32]).forward(x_image)
        # h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 64)
        a2 = CnnLayer([5, 5, 32, 64]).forward(a1)
        # h_pool2(batch, 7, 7, 64) -> h_fc1(1, 1024)
        flat_n = 7 * 7 * 64
        bb = FullLayer([flat_n, 1024], activation='relu').forward(tf.reshape(a2, [-1, flat_n]))
        a3 = tf.nn.dropout(bb, self.keep_prob)
        # 1024 -> 10
        self.y_predict = tf.nn.softmax(FullLayer([1024, 10]).forward(a3))

        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(y_predict))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def feed(self, x, y, keep_prob):
        return {self.x: x, self.y: y, self.keep_prob: keep_prob}

    def train(self, mnist, steps=5000, batch_size=500):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = time.clock()
        for i in range(steps):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                print("step %d, loss %g" % (i,
                                            self.cross_entropy.eval(session=sess, feed_dict=self.feed(batch[0], batch[1], 0.5))))
                self.train_step.run(session=sess,
                                    feed_dict=self.feed(batch[0], batch[1], 0.5))
        test_x, test_y = mnist.test.images, mnist.test.labels
        print("test accuracy %g" % self.accuracy.eval(session=sess,
                                                      feed_dict=self.feed(test_x, test_y, 1.0)))
        end = time.clock()
        print("running time is {} s".format(end - start))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    cnn = CNN()
    cnn.train(mnist)

'''

'''
