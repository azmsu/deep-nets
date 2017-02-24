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
def simple_network(x, W, b):
    '''Takes the vectors x (1x784), b (1x9) and matrix W (784x9) and returns the output vector o (1x9)'''
    o = np.dot(x, W) + b
    return o


