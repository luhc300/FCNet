"""
Created by Haochuan Lu on 6/3/17.
"""
import tensorflow as tf
import numpy as np


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=1e-1)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    initial = tf.constant(1e-2, shape=shape)
    return tf.Variable(initial,name=name)


def L_variable(name,shape=None,init=None):
    if init is not None:
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial,name=name)
    else:
        return tf.convert_to_tensor(init, dtype='float',name=name)


def get_batch(X,y,batch_size):
    N = X.shape[0]
    a = np.arange(N)
    np.random.shuffle(a)
    mask = a[:batch_size]
    #print(mask)
    XX = X[mask]
    yy = y[mask]
    return XX, yy


class FCNet:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims) + 1
        self.trained = False
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
        self.saver = tf.train.Saver()
        #with tf.Session() as sess:
            #sess.run(tf.initialize_all_variables())

    def train(self, X, y, reg=1e-5, learning_rate=1e-7, batch_size=100,trained_model_path = None):
        layer_results = []
        X_input = tf.placeholder("float",shape = [None,self.input_dim])
        y_input = tf.placeholder("float",shape = [None,self.output_dim])
        lr = tf.placeholder("float")
        X0 = tf.convert_to_tensor(X_input)
        layer_results.append(X0)
        reg_tern = tf.convert_to_tensor(0.0)
        for i in range(self.num_layers):
            X_output = tf.nn.relu(tf.matmul(layer_results[i], self.params_W[i]) + self.params_b[i])
            X_output_drop = tf.nn.dropout(X_output,1)
            layer_results.append(X_output_drop)
            reg_tern = reg_tern + tf.reduce_sum(self.params_W[i]*self.params_W[i])
        y_result = tf.nn.softmax(layer_results[self.num_layers])
        cross_entropy = -tf.reduce_sum(y_input * tf.log(y_result), name='cross_entropy')
        loss = cross_entropy #+ reg_tern * reg * 0.5
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        prediction = tf.argmax(y_result, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        cur_learning_rate = learning_rate

        with tf.Session() as sess:
            if trained_model_path is None:
                sess.run(tf.initialize_all_variables())
            if trained_model_path is not None:
                if self.trained == False:
                    sess.run(tf.initialize_all_variables())
                else:
                    sess.run(tf.initialize_all_variables())
                    self.saver.restore(sess,trained_model_path)
                    print('model loaded')


            #print(sess.run(self.params_W[0]))
            for j in range(1500):
                X_batch, y_batch = get_batch(X, y, batch_size)
                if j % 100 == 0:
                    print(sess.run(loss,feed_dict={ X_input: X_batch, y_input:y_batch }))
                    print(sess.run(accuracy,feed_dict={ X_input: X_batch, y_input:y_batch }))
                    #print(sess.run(self.params_W[0]))
                    #print(sess.run(layer_results[-1], feed_dict={X_input: X_batch, y_input: y_batch}))
                    #print(sess.run(y_result,feed_dict={X_input:X_batch}))
                    cur_learning_rate = cur_learning_rate * 0.95
                sess.run(train_step,feed_dict={ X_input: X_batch, y_input:y_batch ,lr: cur_learning_rate*0.95})
            saver_path = self.saver.save(sess, "model/model_3.ckpt")
            print("Model saved in file: ", saver_path)
            #print(sess.run(self.params_W[0]))
        self.trained = True

    def predict(self, X, y=None, model_path=None):

        layer_results = []
        X_input = tf.placeholder("float", shape=[None, self.input_dim])
        y_input = tf.placeholder("float", shape=[None, self.output_dim])
        X0 = tf.convert_to_tensor(X_input)
        layer_results.append(X0)
        for i in range(self.num_layers):
            X_output = tf.nn.relu(tf.matmul(layer_results[i], self.params_W[i]) + self.params_b[i])
            layer_results.append(X_output)
        y_result = tf.nn.softmax(layer_results[self.num_layers])
        prediction = tf.argmax(y_result, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        with tf.Session() as sess:
            if model_path is not None:
                self.saver.restore(sess,model_path)
            #print(sess.run(self.params_b[1]))
            results = sess.run(prediction,feed_dict={X_input:X})
            if y is not None:
                print(sess.run(accuracy,feed_dict={X_input:X,y_input:y}))
        return results



