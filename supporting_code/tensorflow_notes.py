#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:11:14 2017

@author: ChrisErnst
"""

import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
    
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])



with tf.Session() as sess2:
    # Run the tf.constant operation in the session
    output1 = sess2.run(C)
    print(output1)
    

# Placeholders    
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)


def placeHolder(string):
    output=None
    x = tf.placeholder(tf.string)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: string})
    return print(output)
    
placeHolder('Good Morning World!')


# Add in TF
x = tf.add(5, 2)  # 7

# Subtract
x = tf.subtract(10, 4) # 6

# Multiply
y = tf.multiply(2, 5)  # 10

# Cast as the same datatype so no error occurs
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1

# Do some math and print the output:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
    
    
# Put together the y=Wx+b equation:    
matW = tf.constant([[-0.5,0.2,0.1],[0.7,-0.8,0.2]])
matX = tf.constant([  [0.2], [0.5], [0.6]  ])
b = tf.constant([[0.1],[0.2]])
prod = tf.matmul(matW,matX)
sum1 = tf.add(prod,b)

with tf.Session() as sess:
    output = sess.run(sum1)
    print(output)
    

# Variables
x = tf.Variable(5)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# Random weight initialization
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

# Zeros
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))


# Build Linear Classifier:
def get_weights(n_features, n_labels):
    
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))



def get_biases(n_labels):

    return tf.Variable(tf.zeros(n_labels))



# Linear Function (xW + b)
def linear(input, w, b):

    return tf.add(tf.matmul(input,w) , b)


# Softmax Function
x = tf.nn.softmax([2.0, 1.0, 0.2])


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # Calculate the softmax of the logits
    softmax = tf.nn.softmax(logit_data)    
    
    with tf.Session() as sess:
        # Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output


# Cross entropy taking in softmax data and one-hot encoded data

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)


cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))




from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/Users/ChrisErnst/anaconda/envs/py35/lib/python3.5/site-packages/tensorflow/include/unsupported/Eigen/CXX11/src/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Mini Batching to save RAM

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Some examples
# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]


import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches

example_batches = batches(batch_size, example_features, example_labels)


# Hidden Layer with ReLU activation function

hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)



# ReLUs in TensorFlow

import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# Create Model
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# Print session results
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits))
    

# Dropout (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
# During training, a good starting value for keep_prob is 0.5.
# During testing, use a keep_prob value of 1.0 to keep all units and maximize the power of the model.
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])



# Dropout in Practice:
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))









