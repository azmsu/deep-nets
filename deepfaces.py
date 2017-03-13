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
from numpy import *
import os
from pylab import *
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
import urllib
from numpy import random
from caffe_classes import class_names

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
        a_array = []
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
                timeout(testfile.retrieve, (line.split()[4], "part10/"+filename), {}, 30)

                # remove images that are unreadable
                try:
                    imread("part10/"+filename).astype(float32)
                except IOError:
                    if os.path.isfile("part10/"+filename):
                        os.remove("part10/"+filename)
                    print "IOError"
                    continue

                if not os.path.isfile("part10/"+filename):
                    continue

                # remove images that have mismatched sha256 codes
                h = hashlib.sha256()
                h.update(open("part10/"+filename).read())
                # print h.hexdigest(), line.split()[6]
                if str(h.hexdigest()) != line.split()[6]:
                    print "SHA256 does not match"
                    continue

                print filename

                # crop image
                coords = line.split("\t")[4].split(",")
                img = crop("part10/"+filename, coords)
                # imsave("cropped/crop_"+filename, img)

                # resize image to 227 x 227
                try:
                    img = imresize(img, (227, 227))
                except ValueError:
                    os.remove("part10/"+filename)
                    print "ValueError"
                    continue

                # swap red and blue channels
                img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]

                # store image in set
                a_array += [img]

                i += 1

        set[name] = a_array

    return set

def partition(conv4):
    '''
    Partition data set into training, validation and testing sets. The training and validation sets will have 30 images
    of each actor/actress, randomly chosen. The training set will contain the remaining images.
    :param conv4: output from conv4 layer
    :return: x and y matrices for the training, validation, and testing sets. train_set is a set of same format as
             conv4 but containing only the partitioned training data.
    '''
    train_x = zeros((0, 13*13*384))
    train_y_ = zeros((0, 6))
    test_x = zeros((0, 13*13*384))
    test_y_ = zeros((0, 6))
    val_x = zeros((0, 13*13*384))
    val_y_ = zeros((0, 6))
    names = ["drescher", "ferrera", "chenoweth", "baldwin", "hader", "carell"]
    train_set = {}

    for k in range(6):
        size = len(conv4[names[k]])

        random_perm = random.permutation(size)

        idx_test = array(random_perm[:30])
        idx_val = array(random_perm[30:60])
        idx_train = array(random_perm[60:])

        train_set[names[k]] = (array(conv4[names[k]])[idx_train])

        test_x = vstack((test_x, ((array(conv4[names[k]])[idx_test])/255.)))
        val_x = vstack((val_x, ((array(conv4[names[k]])[idx_val])/255.)))
        train_x = vstack((train_x, ((array(conv4[names[k]])[idx_train])/255.)))

        one_hot = zeros(6)
        one_hot[k] = 1

        test_y_ = vstack((test_y_, tile(one_hot, (30, 1))))
        val_y_ = vstack((val_y_, tile(one_hot, (30, 1))))
        train_y_ = vstack((train_y_, tile(one_hot, (size - 60, 1))))

    return train_x, train_y_, test_x, test_y_, val_x, val_y_, train_set

def get_batch(train_set, N):
    '''
    gets a batch for minibatch training from the partitioned training set of size N.
    :param train_set: partitioned trianing set data.
    :param N: size of minibatch.
    :return: minibatch.
    '''
    n = N/6
    batch_x = zeros((0, 13*13*384))
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
    # create placeholder for input (13*13*384 neurons)
    x = tf.placeholder(tf.float32, [None, 13*13*384])

    # randomly initialize the weights and biases for each layer with a normal distribution (standard deviation of 0.01)
    # layer connecting input to hidden
    W0 = tf.Variable(tf.random_normal([13*13*384, nhid], stddev=sdev))
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

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''
    From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

if __name__ == "__main__":
    act = ["Fran Drescher", "America Ferrera", "Kristin Chenoweth", "Alec Baldwin", "Bill Hader", "Steve Carell"]
    names = ["drescher", "ferrera", "chenoweth", "baldwin", "hader", "carell"]

    if not os.path.exists("part10_set.npy"):
        input = create_set(act)
        np.save("part10_set.npy", input)
    else:
        input = np.load("part10_set.npy").item()

    train_x = zeros((1, 227, 227, 3)).astype(float32)
    xdim = train_x.shape[1:]

    net_data = load("bvlc_alexnet.npy").item()

    x = tf.placeholder(tf.float32, (None,) + xdim)

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    if not os.path.exists("conv4_set.npy"):
        conv4_set = {}
        for i in range(6):
            conv4_out = sess.run(conv4, feed_dict={x: input[names[i]]})
            a_array = zeros((0, 13*13*384))
            for j in range(conv4_out.shape[0]):
                flattened = conv4_out[j, :, :, :].reshape([1, 13*13*384])
                a_array = vstack((a_array, flattened))
            conv4_set[names[i]] = a_array

        np.save("conv4_set.npy", conv4_set)
    else:
        conv4_set = np.load("conv4_set.npy").item()

    train_x, train_y_, test_x, test_y_, val_x, val_y_, train_set = partition(conv4_set)

    # PART 10: deep neural network with convolution layer 4 output as input of single-hidden layer network
    N = 180
    minibatch_enabled = True

    act1, act2, nhid, lam = "tanh", "t", 30, 0.00005
    print "The chosen parameters are:"
    print "\tFirst layer activation function:", act1
    print "\tSecond layer (output) activation function:", act2
    print "\tNumber of hidden neurons:", nhid
    print "\tLambda", lam
    print
    sess, x, y_, train_step, decay_penalty, accuracy, W0, W1, layer1 = create_nn(act1, act2, nhid=nhid, sdev=0.01, lam=lam)

    test_plot = ""
    val_plot = ""
    train_plot = ""

    for i in range(501):
        if minibatch_enabled:
            batch_x, batch_y_ = get_batch(train_set, N)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y_})
        else:
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 20 == 0:
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

    # PART 11: visualizing weights
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
        w = reshape(w, [1, 13, 13, 384])

        rand = random.permutation(384)[:5]

        for j in rand:
            wj = w[:, :, :, j]
            wj = wj.reshape([13, 13])

            imsave('part11/p11w'+str(save_name)+'feat'+str(j)+'.jpg', wj)

        x = tf.placeholder(tf.float32, (1, 13, 13, 384))
        k_h = 1; k_w = 1; c_o = 1; s_h = 1; s_w = 1
        convW = tf.Variable(ones((1, 1, 384, 1)).astype(float32))
        convb = tf.Variable(zeros((1,)).astype(float32))
        flat = conv(x, convW, convb, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        flat_out = sess.run(flat, feed_dict={x: w})
        flat_out = flat_out[0, :, :, :]
        flat_out = flat_out.reshape((13, 13))

        imsave('part11/w'+str(save_name)+'.jpg', flat_out)


