#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from functools import partial
import talib
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class model:
    def __init__(self, steps, inputs, nodes, layers, output):
        self.n_steps = steps
        self.n_inputs = inputs
        self.num_nodes = nodes
        self.n_layers = layers
        self.n_outputs = output
    def construct(self, X_test, y_test, X_out_sample, y_out_sample):
        self.X_test = X_test.reshape((-1, self.n_steps, self.n_inputs))
        self.y_test = y_test.reshape((-1, self.n_steps, self.n_outputs))
        self.X_out_sample = X_out_sample.reshape((-1, self.n_steps, self.n_inputs))
        self.y_out_sample = y_out_sample.reshape((-1, self.n_steps, self.n_outputs))
        tf.reset_default_graph()
        tf.set_random_seed(123)
        np.random.seed(123)
        self.he_init2 = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        self.tanh = tf.nn.tanh
        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_steps, self.n_outputs])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())



        self.cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.num_nodes, use_peepholes=True, activation=self.tanh, initializer=self.he_init2) for layer in range(self.n_layers)]
        self.cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in self.cells]
        self.multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(self.cells_drop)


        self.rnn_outputs, self.states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X, dtype=tf.float32)
        self.stacked_rnn_outputs_ = tf.reshape(self.rnn_outputs, [-1, self.num_nodes])
        self.stacked_outputs = tf.layers.dense(self.stacked_rnn_outputs_, self.n_outputs)
        self.outputs = tf.reshape(self.stacked_outputs, [-1, self.n_steps, self.n_outputs])


        self.learning_rate = 0.005


        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))

        #Gradient clip for exploding gradient
        self.threshold = 0.3
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -self.threshold, self.threshold), var)
                      for grad, var in self.grads_and_vars]
        self.training_op = self.optimizer.apply_gradients(self.capped_gvs) # then apply it, normally minimize funtion do it all



        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.train_keep_prob = 0.7
        self.max_checks_without_progress = 40
        self.checks_without_progress = 0
        self.best_loss = np.infty
        self.n_iterations = 200
        self.batch_size = 180
        self.mse_train = []
        self.mse_test = []
        self.mse_out_ = []

        
        
    def shuffle_batch(self, X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X)//batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch
        
    
    def fit(self, X_train, y_train, save_path, return_loss=True):
        with tf.Session() as sess:
            self.init.run()
            for iteration in range(self.n_iterations):
                for X_batch, y_batch in self.shuffle_batch(X_train, y_train, self.batch_size):
                    X_batch = X_batch.reshape((-1, self.n_steps, self.n_inputs))
                    y_batch = y_batch.reshape((-1, self.n_steps, self.n_outputs))
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})
                mse = self.loss.eval(feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob: self.train_keep_prob})
                mse_ = self.loss.eval(feed_dict={self.X: self.X_test, self.y: self.y_test, self.keep_prob: self.train_keep_prob})
                mse_out = self.loss.eval(feed_dict={self.X: self.X_out_sample, self.y: self.y_out_sample})
                self.mse_train.append(mse)
                self.mse_test.append(mse_)
                self.mse_out_.append(mse_out)
                print(iteration, "\tMSE:", mse, " MSE_TEST", mse_, " MSE_out", mse_out)
                if mse_ < self.best_loss:
                  # adding stop loss
                  save_path = self.saver.save(sess, save_path)
                  self.best_loss = mse_
                  self.checks_without_progress = 0
                else:
                  self.checks_without_progress += 1
                  if self.checks_without_progress > self.max_checks_without_progress:
                    print("Early stopping!")
                    break
        
        if return_loss == True:
            return self.mse_train, self.mse_test, self.mse_out_
    
    def predict(self, model_path, X_test, X_out_sample):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            self.y_test_pred = sess.run(self.outputs, feed_dict={self.X: X_test})
            self.y_pred_out = sess.run(self.outputs, feed_dict={self.X: X_out_sample})
        return self.y_test_pred.reshape(-1,), self.y_pred_out.reshape(-1,)
    

    def evalu(self, y_test, y_out_sample):
        print(" MSE-Valid:", mean_squared_error(y_test, self.y_test_pred.reshape(-1,)), "\n", 
             "MSE-Test:", mean_squared_error(y_out_sample, self.y_pred_out.reshape(-1,)), "\n", 
        "R,Squared-Valid:", pd.Series(y_test).corr(pd.Series(self.y_test_pred.reshape(-1,)))**2, "\n", 
        "R,Squared-Test:", pd.Series(y_out_sample).corr(pd.Series(self.y_pred_out.reshape(-1,)))**2)
                    
    def evalu_direct(self, y_test, y_pred):
        self.comparison_change = pd.concat([pd.DataFrame(y_test.reshape((-1, 1))), 
                        pd.DataFrame(y_pred.reshape((-1, 1)))], axis=1)
        self.comparison_change.columns = ["True", "Pred"]
        self.test = []
        for a in self.comparison_change["True"]:
            if a >= 0:
                self.test.append(1)
            else:
                self.test.append(0)

        self.pred = []
        for a in self.comparison_change["Pred"]:
            if a >= 0:
                self.pred.append(1)
            else:
                self.pred.append(0)
        print("ACCR Score:", accuracy_score(self.test, self.pred))
        print(classification_report(self.test, self.pred))

        
        
        
        
        
        

