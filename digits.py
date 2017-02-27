import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


# PART 2
def network(X, W, B):
    '''
    Takes the matrices X, W and B and returns the softmax probabilities matrix P
    :param X: flattened image matrix where each row is a flattened image (nx784 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x784 numpy array)
    :param B: bias matrix where the (ij)th element is the bias for image j's ith output (nx10 numpy array)
    :return: the softmax probabilities matrix (nx10 numpy array)
    '''
    O = np.dot(X, W.T) + B
    P = np.exp(O)/np.sum(np.exp(O), 1)
    return P

# PART 3
def cost(P, Y):
    '''
    Takes the matrices P and Y and computes the cost function
    :param P: probabilities matrix where each row is the softmax probability vector for an image (nx10 numpy array)
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :return: cost
    '''
    C = -1.*np.sum(np.sum(Y*np.log(P), 1), 0)
    return C

def gradient(P, Y, X):
    '''
    Takes the matrices P, Y and X and computes the vectorized gradient of the cost function with respect to the weights,
    for n images
    :param P: probabilities matrix where each row is the softmax probability vector for an image (nx10 numpy array)
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx784 numpy array)
    :return: dC/dW the gradient the cost function C with respect to the weights (10x784 numpy array)
    '''
    grad = np.dot((P - Y).T, X)
    return grad

def finite_diff(Y, X, W, B, h):
    '''
    Take the matrices Y, X, W and B and computes the finite difference approximation of the gradient with respect to
    the weights
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx784 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x784 numpy array)
    :param B: bias matrix where the (ij)th element is the bias for image j's ith output (nx10 numpy array)
    :param h: step size
    :return: finite difference approximation of gradient with respect to weights (10x784 numpy array)
    '''
    grad = np.zeros((10, 784))
    for i in range(10):
        for j in range(784):
            H = np.zeros((10, 784))
            H[i, j] = h
            P2 = network(X + H, W, B)
            P1 = network(X, W, B)
            grad[i, j] = (cost(P2, Y) - cost(P1, Y))/(h*1.)
    return grad

# PART 4
def train():
    return


