#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:16:54 2017

@author: ChrisErnst
"""

import numpy as np

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

# Define some of our heavy lifting functions
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
def prediction(X, W, b):
    return sigmoid(np.matmul(X,W)+b)
def error_vector(y, y_hat):
    return [-y[i]*np.log(y_hat[i]) - (1-y[i])*np.log(1-y_hat[i]) for i in range(len(y))]
def error(y, y_hat):
    ev = error_vector(y, y_hat)
    return sum(ev)/len(ev)

# Example Data: (taken from perception.csv) - first 5 rows, last 5 rows
data = [0.78051,-0.063669,1,
0.28774,0.29139,1,
0.40714,0.17878,1,
0.2923,0.4217,1,
0.50922,0.35256,1,
0.77029,0.7014,0,
0.73156,0.71782,0,
0.44556,0.57991,0,
0.85275,0.85987,0,
0.51912,0.62359,0]

data = np.reshape(data, [10,3])

# input data
X = data[:,0:2]
X1 = X[:,0]
X2 = X[:,1]

# output(1 or 0)
Y = data[:,2]

# define initial weights- only temporarily, as they are randomized later on...
W = np.ones([np.size(X,axis=1), np.size(X,axis=0)])
W1 = W[0,:]
W2 = W[1,:]

# Bias unit
bias = 1

# A single prediction for the first observation is:
sigmoid(data[0,0]*data[0,1] + bias)

# In loop form this is:
y_hat=[]
for i in range(len(data)):
    y_hat.append(sigmoid(data[i,0]*data[i,1] + bias))

# Add the predicted values onto the end of the matrix
newData = np.insert(data, 3, y_hat, axis=1)



# The code below calculates the gradient of the error function.
# The result should be a list of three lists:
    
# The first list should contain the gradient (partial derivatives) with respect to w1
# The second list should contain the gradient (partial derivatives) with respect to w2
# The third list should contain the gradient (partial derivatives) with respect to b

def dErrors(X, y, y_hat):
    DErrorsDx1 = [X[i][0]*(y[i]-y_hat[i]) for i in range(len(y))]
    DErrorsDx2 = [X[i][1]*(y[i]-y_hat[i]) for i in range(len(y))]
    DErrorsDb = [y[i]-y_hat[i] for i in range(len(y))]
    return DErrorsDx1, DErrorsDx2, DErrorsDb



Dx1, Dx2, Dxb = dErrors(X,Y,y_hat)



# The code below implements the gradient descent step.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b.
# It should calculate the prediction, the gradients, and use them to
# update the weights and bias W, b. Then return W and b.
# The error e will be calculated and returned for you, for plotting purposes.
def gradientDescentStep(X, y, W, b, learn_rate = 0.05):
    y_hat = prediction(X,W,b)
    errors = error_vector(y, y_hat)
    derivErrors = dErrors(X, y, y_hat)
    W[0] += sum(derivErrors[0])*learn_rate
    W[1] += sum(derivErrors[1])*learn_rate
    b += sum(derivErrors[2])*learn_rate
    return W, b, sum(errors)


W, b, totalError = gradientDescentStep(X,Y, W, bias)

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainLR(X, y, learn_rate = 0.05, num_epochs = 1000):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    # Initialize the weights randomly
    W = np.array(np.random.rand(2,1))*2 -1
    b = np.random.rand(1)[0]*2 - 1
    # These are the solution lines that get plotted below.
    boundary_lines = []
    errors = []
    for i in range(num_epochs):
        # In each epoch, we apply the gradient descent step.
        W, b, error = gradientDescentStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        errors.append(error)
    return boundary_lines, errors


boundary_lines, errors = trainLR(X,Y)
