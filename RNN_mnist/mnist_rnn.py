#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:58:44 2016

@author: charl
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./", one_hot=True)

chunk_size=28
chunk_n=28

rnn_size=256
n_output_layer = 10

X = tf.placeholder(tf.float32, [None, chunk_n, chunk_size])
Y = tf.placeholder(tf.float32)

def recuurent_neural_network(data):
    layer = {"w_": tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             "b_": tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(value=data, num_or_size_splits=chunk_n, axis=0)

    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    return output

batch_size = 100

def train_neural_network(X, Y):
    predict = recuurent_neural_network(X)
    cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_fun)
    
    epochs = 13
    losses=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(len(mnist.train.labels)/batch_size )):
                x,y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, chunk_n, chunk_size])
                _, c = sess.run([optimizer, cost_fun], feed_dict={X:x, Y:y})
#                print(c)
                epoch_loss += c
            print(epoch," : ", epoch_loss)
            losses.append(epoch_loss)
        corrent = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrent, tf.float32))
        print("accuracy: {}".format(accuracy.eval({X:mnist.test.images.reshape(-1, chunk_n, chunk_size), Y:mnist.test.labels})))
        
        
        
train_neural_network(X, Y)