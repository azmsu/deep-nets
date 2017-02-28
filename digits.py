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
from scipy.io import loadmat

# np.set_printoptions(threshold=1000000)

# FUNCTIONS
def get_data(set):
    '''
    Gets data from set and returns the data in the matrices X and Y
    :param set: 'train' or 'test'
    :return: X and Y, respectively the image matrix and the classifications matrix
    '''
    # Load the MNIST digit data
    M = loadmat('mnist_all.mat')

    X = None
    Y = None
    for i in range(10):
        x = M[set + str(i)]/255.0
        rows = x.shape[0]
        x = np.hstack((np.array([np.ones(rows)]).T, x))
        y = np.zeros((rows, 10))
        # set the ith column of y to 1 (one hot encoding)
        y[:, i] = np.ones(rows)
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
        if Y is None:
            Y = y
        else:
            Y = np.vstack((Y, y))

    return X, Y

def get_performance(Y, X, W):
    '''
    Gets the performance of the (test) data given the W matrix
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :return: percentage of correct classifications
    '''
    P = softmax(X, W)
    indices_test = np.argmax(P, 1)
    indices_actual = np.argmax(Y, 1)
    performance = (indices_test.shape[0] - np.count_nonzero(indices_test - indices_actual))/(.01*indices_test.shape[0])
    return performance

# PART 2
def softmax(X, W):
    '''
    Takes the matrices X and W and returns the softmax probabilities matrix P
    :param X: flattened image matrix where each row is a flattened image with a bias (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :return: the softmax probabilities matrix (nx10 numpy array)
    '''
    O = np.dot(X, W.T)
    P = np.exp(O)/(np.array([np.sum(np.exp(O), 1)])).T
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
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :return: dC/dW the gradient the cost function C with respect to the weights (10x785 numpy array)
    '''
    grad = np.dot((P - Y).T, X)
    return grad

def finite_diff(Y, X, W, h):
    '''
    Take the matrices Y, X and W and computes the finite difference approximation of the gradient with respect to
    the weights
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :param h: step size
    :return: finite difference approximation of gradient with respect to weights (10x784 numpy array)
    '''
    grad = np.zeros((10, 785))
    P1 = softmax(X, W)
    c1 = cost(P1, Y)
    for i in range(10):
        print i
        for j in range(785):
            H = np.zeros((10, 785))
            H[i, j] = h
            P2 = softmax(X, W + H)
            grad[i, j] = (cost(P2, Y) - c1)/(h*1.)
    return grad

# PART 4
def grad_descent(Y, X, W, alpha):
    '''
    Executes gradient descent on the data set; computes the W matrix for which the cost function is minimized
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :param alpha: step size for gradient descent
    :return: W matrix for which cost is minimized
    '''
    EPS = 1e-6
    prev_W = W - 10*EPS
    max_iter = 5000
    iter = 0

    X_test, Y_test = get_data('test')

    while np.linalg.norm(W-prev_W) > EPS and iter < max_iter:
        prev_W = W.copy()
        P = softmax(X, W)
        W -= alpha*gradient(P, Y, X)
        if iter % 100 == 0:
            print 'Iteration:', iter
            print 'Cost', cost(P, Y)
            print 'Performance', get_performance(Y_test, X_test, W)
            print
        iter += 1

    c = cost(P, Y)

    print "Minimum found at", W, "with cost function value of", c, "on iteration", iter
    return W

# MAIN CODE
X, Y = get_data('train')
W = np.ones((10, 785))

# PART 3 - test gradient
# print 'running part 3...'
# fd_grad = finite_diff(Y, X, W, 1e-3)
# print fd_grad
# P = softmax(X, W)
# grad = gradient(P, Y, X)
# print grad
#
# print np.norm(grad - fd_grad)
# print 'done part 3'

# PART 4 - train network
print 'running part 4...'
W = grad_descent(Y, X, W, 1e-5)
print 'done part 4'

