"""
Created by Haochuan Lu on 6/3/17.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as ra_score
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
    '''
    N = X.shape[0]
    a = np.arange(N)
    np.random.shuffle(a)
    mask = a[:batch_size]
    #print(mask)
    XX = X[mask]
    yy = y[mask]
    return XX, yy
    '''

    default_1 = np.where(y == 1)[0]
    default_0 = np.where(y == 0)[0]
    np.random.shuffle(default_0)
    np.random.shuffle(default_1)
    random_default_0 = default_0[:batch_size]
    random_default_1 = default_1[:batch_size]
    # print(random_default_0)
    under_sample_default_0 = X[random_default_0]
    under_sample_default_1 = X[random_default_1]
    y_0 = y[random_default_0]
    y_1 = y[random_default_1]
    under_sample_X = np.concatenate([under_sample_default_0, under_sample_default_1])
    under_sample_y = np.concatenate([y_0,y_1])
    # print(len(under_sample_indices))
    y_dummies = np.array(pd.get_dummies(under_sample_y))
    #print(y_dummies)
    return under_sample_X, y_dummies

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

    def train(self, X, y, reg=1e-7, learning_rate=1e-7, batch_size=100,trained_model_path = None):
        layer_results = []
        X_input = tf.placeholder("float",shape = [None,self.input_dim])
        y_input = tf.placeholder("float",shape = [None,self.output_dim])
        lr = tf.placeholder("float")
        X0 = tf.convert_to_tensor(X_input)
        layer_results.append(X0)
        reg_tern = tf.convert_to_tensor(reg)
        for i in range(self.num_layers-1):
            X_output = tf.nn.relu(tf.matmul(layer_results[i], self.params_W[i]) + self.params_b[i])
            X_output_drop = tf.nn.dropout(X_output,1)

            layer_results.append(X_output_drop)

            reg_tern = reg_tern + tf.reduce_sum(self.params_W[i]*self.params_W[i])
        y_result = tf.nn.softmax(tf.matmul(layer_results[self.num_layers-1],self.params_W[self.num_layers - 1]) + self.params_b[self.num_layers - 1])
        cross_entropy = -tf.reduce_sum(y_input * tf.log(y_result + 1e-10), name='cross_entropy')
        loss = cross_entropy + reg_tern * reg * 0.5
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        prediction = tf.argmax(y_result, 1)
        ori_prediction = tf.argmax(y_input, 1)
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
            for j in range(120000):
                X_batch, y_batch = get_batch(X, y, batch_size)
                if j % 100 == 0:
                    print("iter %d" %j)
                    print(sess.run(loss,feed_dict={ X_input: X_batch, y_input:y_batch }))
                    print(sess.run(accuracy,feed_dict={ X_input: X_batch, y_input:y_batch }))
                    results = sess.run(prediction, feed_dict={X_input: X_batch, y_input:y_batch})
                    ori_result = sess.run(ori_prediction, feed_dict={X_input: X, y_input:y_batch})
                    #print(results.shape)
                    rascore = ra_score(ori_result, results)
                    print("ROC_AUC %f" %rascore)
                    #print(sess.run(self.params_W[0]))
                    #print(sess.run(layer_results[-1], feed_dict={X_input: X_batch, y_input: y_batch}))
                    #print(sess.run(y_result,feed_dict={X_input:X_batch}))
                    cur_learning_rate = cur_learning_rate * 1
                sess.run(train_step,feed_dict={ X_input: X_batch, y_input:y_batch ,lr: cur_learning_rate})
            saver_path = self.saver.save(sess, trained_model_path)
            print("Model saved in file: ", saver_path)
            #print(sess.run(self.params_W[0]))
        self.trained = True

    def predict(self, X, y=None, model_path=None):

        layer_results = []
        X_input = tf.placeholder("float", shape=[None, self.input_dim])
        y_input = tf.placeholder("float", shape=[None, self.output_dim])
        X0 = tf.convert_to_tensor(X_input)
        layer_results.append(X0)
        for i in range(self.num_layers - 1):
            X_output = tf.nn.relu(tf.matmul(layer_results[i], self.params_W[i]) + self.params_b[i])
            layer_results.append(X_output)
        y_result = tf.nn.softmax(tf.matmul(layer_results[self.num_layers-1],self.params_W[self.num_layers - 1]) + self.params_b[self.num_layers - 1])
        prediction = y_result[:,1]
        #correct_prediction = tf.equal(prediction, tf.argmax(y_input, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        with tf.Session() as sess:
            if model_path is not None:
                self.saver.restore(sess,model_path)
            #print(sess.run(self.params_b[1]))
            results = sess.run(prediction,feed_dict={X_input:X})
            if y is not None:
                pass
                #print(sess.run(accuracy,feed_dict={X_input:X,y_input:y}))
        return results



