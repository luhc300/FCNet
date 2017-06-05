"""
Created by Haochuan Lu on 6/3/17.
"""
import tensorflow as tf
import numpy as np


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def L_variable(name,shape=None,init=None):
    if init is not None:
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial,name=name)
    else:
        return tf.convert_to_tensor(init, dtype='float',name=name)


def get_batch(X,y,batch_size):
    N = X.shape[0]
    mask = np.random.choice(N,batch_size)
    XX = X[mask]
    yy = y[mask]
    return XX, yy


class FCNet:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims) + 1
        self._prepare()

    def _prepare(self):
        self.params_W = []
        self.params_b = []
        layer_dim = []
        layer_dim.append(self.input_dim)
        layer_dim.extend(self.hidden_dims)
        layer_dim.append(self.output_dim)
        for i in range(self.num_layers):
            W = weight_variable([layer_dim[i], layer_dim[i+1]], name='W'+str(i+1))
            b = bias_variable([layer_dim[i+1]],name='b'+str(i+1))
            self.params_W.append(W)
            self.params_b.append(b)

    def train(self, X, y, reg=1e-4, learning_rate=1e-4, batch_size=100):
        layer_results = []
        X_input = tf.placeholder("float",shape = [None,self.input_dim])
        y_input = tf.placeholder("float",shape = [None,self.output_dim])
        X0 = X_input
        layer_results.append(X0)
        for i in range(self.num_layers):
            X_output = tf.nn.relu(tf.matmul(layer_results[i], weight) + bias)
        update = tf.assign(X_stage,X_output,validate_shape=False)
        y_result = tf.nn.softmax(X_stage)
        cross_entropy = -tf.reduce_sum(y * tf.log(y_result), name='cross_entropy')
        train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
        with tf.Session() as sess:
            sess.run(X_stage)
            for j in range(20000):
                X_batch, y_batch = get_batch(X, y, batch_size)
                X_stage = X_variable('X_stage',init=X_batch)
                print(sess.run(X_stage))
                for i in range(self.num_layers):
                    sess.run(update,feed_dict={X_input: X_stage, weight: self.params_W[i], bias: self.params_b[i]})
                sess.run(train_step)
                if j % 100 == 0:
                    print(sess.run(cross_entropy))

