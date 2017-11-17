#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:18:35 2017

@author: ChrisErnst

CNN Dimensions
"""

import tensorflow as tf

# Padding
P=1

# Stride
S=2

# Number of filters
num_filters  = 20 # 20 of 8x8

input_height = 32
input_width = 32

filter_height = 8
filter_width = 8

new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1
new_depth =  num_filters



# Done with TensorFlow:
    
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias