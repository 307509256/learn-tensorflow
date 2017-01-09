#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:18:56 2017

@author: charl
"""

from __future__ import print_function 

import collections
import tensorflow as tf
import numpy as np

poetry_file='poetry.txt'

poetrys = []

with open(poetry_file,"r") as f:
    lines = f.readlines()
    for line in lines:
        try:
            title, content = line.strip().split(":")
            content = content.replace(' ','')
            if '_' in content or '(' in content or '《' in content:
                continue
            if len(content) < 5 or len(content) > 79:
			continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass
        
#sorted

poetrys = sorted(poetrys, key=lambda line:len(line))
print("Numbers of the poetrys", len(poetrys))
#Numbers of each words
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
counter_pairs = sorted(counter.items(), key=lambda x:-x[1])
words,_ = zip(*counter_pairs)
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]


batch_size = 128
n_block = len(poetrys_vector)//batch_size
x_batch=[]
y_batch=[]

for i in range(n_block):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batches = poetrys_vector[start_index:end_index]
    length = max(map(len,batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]
    x_batch.append(xdata)
    y_batch.append(ydata)


###Char RNN


input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

def neural_network(model='lstm', rnn_size=128, num_layer=2):
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layer, state_is_tuple=True)
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    with tf.VariableScope("rnnlm"):
        softmax_w = tf.get_variable("softmax_w",[rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b",[len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",[len(words)+1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, inputs)
        




































