#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:34:44 2017

@author: charl
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def activation(layer):
    return tf.nn.relu(layer)
def convolutional(input_layer,w,b):
    return tf.add(tf.nn.conv2d(input_layer, w, strides=[1,1,1,1],padding='SAME'), b)
    
keep_prob = tf.placeholder(data_type())

def convolutional_neural_network(data):

    global keep_prob
    n_output_layer = 10
    weights = {'w_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
              'w_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
              'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
              'out': tf.Variable(tf.random_normal([1024, n_output_layer]))}
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_output_layer]))}
    data = tf.reshape(data, [-1,28,28,1])
    
    conv1 = activation(convolutional(data,weights['w_conv1'], biases['b_conv1']))
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2 = activation(convolutional(conv1,weights['w_conv2'], biases['b_conv2']))
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    fc = tf.reshape(conv2, [-1,7*7*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc']))
    fc = tf.nn.dropout(fc, keep_prob)
    output = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return output
    
def main(_):
    batch_size = 512
    global keep_prob
    X = tf.placeholder(data_type(), [None, 28*28]) 
    Y = tf.placeholder(data_type())
    
    predict = convolutional_neural_network(X)
    cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_fun)
    mnist = input_data.read_data_sets("./", one_hot=True)
    epochs = 20
    losses=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(len(mnist.train.labels)/batch_size )):
                x,y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost_fun], feed_dict={X:x, Y:y, keep_prob:0.6})
                epoch_loss += c
            print("epoch :" , epoch  ,  "--- loss: " , epoch_loss)
            losses.append(epoch_loss)
        corrent = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrent, tf.float32))
        print("accuracy: {}".format(accuracy.eval({X:mnist.test.images, Y:mnist.test.labels,keep_prob:1})))

        
if __name__ == "__main__":
  tf.app.run()