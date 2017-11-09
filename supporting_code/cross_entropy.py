#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:04:55 2017

@author: ChrisErnst

Cross Entropy helps define the accuracy of a model and the likelihood of the specific
arrangement occuring. It is the equivalent of the AND in probability: P(A) AND P(B)= P(A) * P(B)


"""

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

P=[0.8,0.7,0.1]
Y=[1,1,0]

def cross_entropy(Y, P):

    # Remember, cross entropy is the sum of the negative natural logarithms(probabilities)
    # Y is the list indicating if the outcome or not (0 or 1)
    # P is the probability of the outcome happening
    
    Y = np.float_(Y)
    P = np.float_(P)
    
    return np.sum(Y * np.log(P) + (1 - Y)* np.log(1-P)) * -1
    
    
    
    

    