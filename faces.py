from pylab import *
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
import cPickle
import os
from scipy.io import loadmat
import tensorflow as tf
import hashlib
from scipy.misc import imsave

# generate random seed based on current time
t = int(time.time())
print "t=", t
random.seed(t)

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''
    From: http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def crop(img, coords):
    '''
    Get cropped RGB image.
    :param img: image as numpy array (n x m x 3).
    :param coords: coordinates of crop location of the form [x1, y1, x2, y2], where (x1, y1) is the top-left pixel and
                   (x2, y2) is the bottom right pixel.
    :return: cropped RGB image.
    '''
    img_array = imread(img, mode="RGB")
    return img_array[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])]

def rgb2gray(rgb):
    '''
    Convert RGB image to grayscale image.
    :param rgb: image as numpy array (n x m x 3).
    :return: grayscale image.
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.

def create_set(act):
    '''
    Create set data set containing all valid images of actors/actresses in act.
    :param act: list of actors/actresses' names.
    :return: data set of the form:
                {
                actor1: [[img1] ... [imgn]],
                actor2: [[img1] ... [imgn]],
                   :
                actor6: [[img1] ... [imgn]]
                }
    '''
    testfile = urllib.URLopener()
    set = {}

    # loop through each actor/actress in act
    for a in act:
        a_array = zeros((0, 1024))
        name = a.split()[1].lower()
        i = 0

        # loop through each line in the raw data (every image of every
        # actor/actress is in 'faces_subset.txt'
        for line in open("faces_subset.txt"):
            if a in line:
                # filename is of the form: '[name][###].[ext]'
                filename = name+str(i).zfill(3)+"."+line.split()[4].split(".")[-1]
                # A version without timeout (uncomment in case you need to
                # unsupress exceptions, which timeout() does)
                # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                # timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)

                # remove images that are unreadable
                try:
                    imread("uncropped/"+filename)
                except IOError:
                    if os.path.isfile("uncropped/"+filename):
                        os.remove("uncropped/"+filename)
                    print "IOError"
                    continue

                if not os.path.isfile("uncropped/"+filename):
                    continue

                # remove images that have mismatched sha256 codes
                h = hashlib.sha256()
                h.update(open("uncropped/"+filename).read())
                print h.hexdigest(), line.split()[6]
                if str(h.hexdigest()) != line.split()[6]:
                    print "SHA256 does not match"
                    continue

                print filename

                # crop image
                coords = line.split("\t")[4].split(",")
                img = crop("uncropped/"+filename, coords)
                # imsave("cropped/crop_"+filename, img)

                # convert image to grayscale
                img = rgb2gray(img)
                # imsave("gray/gray_"+filename, img)

                # resize image to 32 x 32
                try:
                    img = imresize(img, (32,32))
                except ValueError:
                    os.remove("uncropped/"+filename)
                    print "ValueError"
                    continue

                # reshape image to 1 x 1024
                img = img.reshape((1, 1024))

                # store image in set
                a_array = vstack((a_array, img))

                i += 1

        set[name] = a_array

    return set

def partition(A):
    '''
    Partition data set into training, validation and testing sets. The training and validation sets will have 30 images
    of each actor/actress, randomly chosen. The training set will contain the remaining images.
    :param A: data set generated from create_set()
    :return: x and y matrices for the training, validation, and testing sets. train_set is a set of same format as
             A but containing only the partitioned training data.
    '''
    train_x = zeros((0, 32*32))
    train_y_ = zeros((0, 6))
    test_x = zeros((0, 32*32))
    test_y_ = zeros((0, 6))
    val_x = zeros((0, 32*32))
    val_y_ = zeros((0, 6))
    names = ["drescher", "ferrera", "chenoweth", "baldwin", "hader", "carell"]
    train_set = {}

    for k in range(6):
        size = len(A[names[k]])

        random_perm = random.permutation(size)

        idx_test = array(random_perm[:30])
        idx_val = array(random_perm[30:60])
        idx_train = array(random_perm[60:])

        train_set[names[k]] = (array(A[names[k]])[idx_train])

        test_x = vstack((test_x, ((array(A[names[k]])[idx_test])/255.)))
        val_x = vstack((val_x, ((array(A[names[k]])[idx_val])/255.)))
        train_x = vstack((train_x, ((array(A[names[k]])[idx_train])/255.)))

        one_hot = zeros(6)
        one_hot[k] = 1

        test_y_ = vstack((test_y_,   tile(one_hot, (30, 1))))
        val_y_ = vstack((val_y_,   tile(one_hot, (30, 1))))
        train_y_ = vstack((train_y_,   tile(one_hot, (size - 60, 1))))

    return train_x, train_y_, test_x, test_y_, val_x, val_y_, train_set

def get_batch(train_set, N):
    '''
    Gets a batch for minibatch training from the partitioned training set of size N.
    :param train_set: partitioned trianing set data.
    :param N: size of minibatch.
    :return: minibatch.
    '''
    n = N/6
    batch_x = zeros((0, 32*32))
    batch_y_ = zeros((0, 6))

    names = ["drescher", "ferrera", "chenoweth", "baldwin", "hader", "carell"]

    for k in range(6):
        train_size = len(train_set[names[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_x = vstack((batch_x, ((array(train_set[names[k]])[idx])/255.)))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_ = vstack((batch_y_, tile(one_hot, (n, 1))))
    return batch_x, batch_y_

def create_nn(act1, act2, nhid = 300, sdev = 0.01, lam = 0.0):
    '''
    Creates the neural network given the input parameters.
    :param act1: activation function for first layer
    :param act2: activation function for second layer (output layer)
    :param nhid: number of hidden neurons
    :param sdev: standard deviation of weight initialization
    :param lam: lambda value for regularization
    :return: neural network (required variables/parameters for outer scope computations)
    '''
    # create placeholder for input (1024 neurons)
    x = tf.placeholder(tf.float32, [None, 1024])

    # randomly initialize the weights and biases for each layer with a normal distribution (standard deviation of 0.01)
    # layer connecting input to hidden
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=sdev))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=sdev))

    # layer connecting hidden to output
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=sdev))
    b1 = tf.Variable(tf.random_normal([6], stddev=sdev))

    # initialize layers with activation functions
    if act1 == "t":
        layer1 = tf.matmul(x, W0)+b0
    elif act1 == "tanh":
        layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    elif act1 == "relu":
        layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)

    if act2 == "t":
        layer2 = tf.matmul(layer1, W1)+b1
    elif act2 == "tanh":
        layer2 = tf.nn.tanh(tf.matmul(layer1, W1)+b1)
    elif act2 == "relu":
        layer2 = tf.nn.relu(tf.matmul(layer1, W1)+b1)

    # softmax output layer
    y = tf.nn.softmax(layer2)

    # create placeholder for classification input (6 neurons)
    y_ = tf.placeholder(tf.float32, [None, 6])

    # define cost and training step
    decay_penalty = lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1

def get_best_nn(act1s, act2s, nhids, lams):
    '''
    Loops through different neural networks and tests on the validation set. Returns the best performing network
    parameters.
    :param act1s: list of activation functions for first layer to test
    :param act2s: list of activation functions for second layer to test
    :param nhids: list of number of hidden neurons to test
    :param lams: list of lambda values to test
    :return: best combination of parameters
    '''
    max_acc = 0
    for act1 in act1s:
        for act2 in act2s:
            for nhid in nhids:
                for lam in lams:
                    max_val_acc = 0
                    sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1 = create_nn(act1, act2, nhid=nhid, sdev=0.01, lam=lam)
                    print "Testing with:"
                    print "\tFirst layer activation function:", act1
                    print "\tSecond layer (output) activation function:", act2
                    print "\tNumber of hidden neurons:", nhid
                    print "\tLambda", lam
                    for i in range(2000):
                        if minibatch_enabled:
                            batch_x, batch_y_ = get_batch(train_set, N)
                            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y_})
                        else:
                            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

                        val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
                        if val_acc > max_val_acc:
                            max_val_acc = val_acc

                    if max_val_acc > max_acc:
                        max_acc = max_val_acc
                        params = [act1, act2, nhid, lam]
                    print "Best validation performance was:", max_val_acc
                    print

    print "The chosen parameters are:"
    print "\tFirst layer activation function:", act1
    print "\tSecond layer (output) activation function:", act2
    print "\tNumber of hidden neurons:", nhid
    print "\tLambda", lam
    print

    return params

if __name__ == "__main__":
    act = ["Fran Drescher", "America Ferrera", "Kristin Chenoweth", "Alec Baldwin", "Bill Hader", "Steve Carell"]
    names = ["drescher", "ferrera", "chenoweth", "baldwin", "hader", "carell"]

    if not os.path.exists("act_set.npy"):
        A = create_set(act)
        np.save("act_set.npy", A)
    else:
        A = np.load("act_set.npy").item()

    train_x, train_y_, test_x, test_y_, val_x, val_y_, train_set = partition(A)


    # PART 7: choosing best network parameters and training/testing on best
    N = 180
    minibatch_enabled = True
    val_test = False  # set True for Part 7

    if val_test:
        # test on validation set and check which set of parameters are the best
        act1s = ["relu", "tanh", "t"]
        act2s = ["relu", "tanh", "t"]
        nhids = [30, 70, 100, 300]
        lams = [0.00000, 0.00005, 0.0001, 0.00015]

        params = get_best_nn(act1s, act2s, nhids, lams)

    else:
        params = ["tanh", "t", 30, 0.00005]

    # now use the best results from the validation set to test on the rest of the sets
    act1, act2, nhid, lam = params[0], params[1], params[2], params[3]
    sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1 = create_nn(act1, act2, nhid=nhid, sdev=0.01, lam=lam)

    test_plot = ""
    val_plot = ""
    train_plot = ""

    for i in range(0):
        if minibatch_enabled:
            batch_x, batch_y_ = get_batch(train_set, N)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y_})
        else:
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 50 == 0:
            print "i=", i

            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
            print "Test:", test_acc
            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            print "Validation:", val_acc
            train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y_})
            print "Train:", train_acc
            print "Penalty:", sess.run(decay_penalty)

            test_plot += str((i, test_acc*100))
            val_plot += str((i, val_acc*100))
            train_plot += str((i, train_acc*100))

    print
    print "Output for LaTeX plotting:"
    print "Test", test_plot
    print "Validation", val_plot
    print "Train", train_plot


    # PART 8: use a small training set to test performance (5 images of each actor) with and without regularization
    train_x, train_y_ = get_batch(train_set, 30)
    minibatch_enabled = False

    # no regularization
    params = ["tanh", "t", 30, 0.00000]

    act1, act2, nhid, lam = params[0], params[1], params[2], params[3]
    sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1 = create_nn(act1, act2, nhid=nhid, sdev=0.01, lam=lam)

    test_plot = ""
    val_plot = ""
    train_plot = ""

    print "NO REGULARIZATION"

    for i in range(3001):
        if minibatch_enabled:
            batch_x, batch_y_ = get_batch(train_set, N)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y_})
        else:
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 50 == 0:
            print "i=", i

            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
            print "Test:", test_acc
            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            print "Validation:", val_acc
            train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y_})
            print "Train:", train_acc
            print "Penalty:", sess.run(decay_penalty)

            test_plot += str((i, test_acc*100))
            val_plot += str((i, val_acc*100))
            train_plot += str((i, train_acc*100))

    print
    print "Output for LaTeX plotting:"
    print "Test", test_plot
    print "Validation", val_plot
    print "Train", train_plot

    # with regularization
    params = ["tanh", "t", 30, 1.5]

    act1, act2, nhid, lam = params[0], params[1], params[2], params[3]
    sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1 = create_nn(act1, act2, nhid=nhid, sdev=0.01, lam=lam)

    test_plot = ""
    val_plot = ""
    train_plot = ""

    print "REGULARIZATION WITH LAMBDA =", lam

    for i in range(3001):
        if minibatch_enabled:
            batch_x, batch_y_ = get_batch(train_set, N)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y_})
        else:
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 50 == 0:
            print "i=", i

            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
            print "Test:", test_acc
            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            print "Validation:", val_acc
            train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y_})
            print "Train:", train_acc
            print "Penalty:", sess.run(decay_penalty)

            test_plot += str((i, test_acc*100))
            val_plot += str((i, val_acc*100))
            train_plot += str((i, train_acc*100))

    print
    print "Output for LaTeX plotting:"
    print "Test", test_plot
    print "Validation", val_plot
    print "Train", train_plot


    # PART 9: visualizing weights
    neurons, neuron_ids = [], []

    # get 1 image of each actor/actress and input to neural network
    batch_x, batch_y_ = get_batch(train_set, 6)
    for i in range(6):
        single_x = np.array([batch_x[i, :]])
        single_y_ = np.array([batch_y_[i, :]])

        acc = sess.run(accuracy, feed_dict={x: single_x, y_: single_y_})

        # image is correctly classified
        if acc == 1:
            # get the output from the hidden layer
            hidden_layer = sess.run(layer1, feed_dict={x: single_x, y_: single_y_})
            # add the index of the max hidden layer neuron
            neurons += [argmax(hidden_layer)]
            # stores which actor/actress does this image belong to
            neuron_ids += [argmax(single_y_)]

    W = sess.run(W0)
    for i in range(len(neurons)):
        w = W[:, neurons[i]]
        a = act[neuron_ids[i]]
        save_name = names[neuron_ids[i]]
        w = reshape(w, [32, 32])
        imsave('weights/w'+str(save_name)+'.jpg', w)
