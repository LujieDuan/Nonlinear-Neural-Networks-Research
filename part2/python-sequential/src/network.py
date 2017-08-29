"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Copyright to Michael Nielsen
Modified By Lujie Duan 2017-08-28
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

from copy import deepcopy

import pickle

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Exchange the following two parts to random initializa parameters
        ##########
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # self.weights = [np.random.randn(y, x)
        #                 for x, y in zip(sizes[:-1], sizes[1:])]
        # Include this line if need to save the parameters
        # pickle_param(self.biases, self.weights)
        ###
        self.biases, self.weights = load_param()
        # Include this line if need to print the parameters for Scala app to use
        # print ["%.55f" % i for i in self.unroll(self.biases, self.weights)]
        #########

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data_vectorized, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n_validation = len(validation_data)
        n_test = len(test_data)
        n = len(training_data_vectorized)
        start_time = time.time()
        for j in xrange(epochs):
            # Include this line if want to shuffle the samples before each epoch
            # random.shuffle(training_data_vectorized)
            mini_batches = [
                training_data_vectorized[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            training_eva = self.evaluate(training_data)
            validation_eva = self.evaluate(validation_data)
            test_eva = self.evaluate(test_data)

            print "Epoch {0}: Training: {1} / {2}, {3}%".format(
                j, training_eva, n, 100.0*training_eva/n)
            print "           Validation: {1} / {2}, {3}%".format(
                j, validation_eva, n_validation, 100.0*validation_eva/n_validation)
            print "           Test: {1} / {2}, {3}%".format(
                j, test_eva, n_test, 100.0*test_eva/n_test)
        print "Total Time: {0}s ".format((time.time() - start_time))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # numb, numw = self.gradient_checking(x, y)
            # gradflat = np.array(self.unroll(delta_nabla_b, delta_nabla_w))
            # numflat = np.array(self.unroll(numb, numw))
            # diff = np.linalg.norm(numflat - gradflat) / np.linalg.norm(numflat + gradflat)
            # print diff
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


    # def feedforward_checking(self, a, y, bias, weights):
    #     """Return the output of the network if ``a`` is input."""
    #     for b, w in zip(bias, weights):
    #         a = sigmoid(np.dot(w, a)+b)
    #     return sum(np.power(a - y, 2)) / 2.0
    #
    # def gradient_checking(self, x, y):
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #
    #     b0 = deepcopy(self.biases)
    #     b1 = deepcopy(self.biases)
    #
    #     ebl = 0.0001
    #
    #     for i in xrange(0, len(self.biases)):
    #         for j in xrange(0, len(self.biases[i])):
    #             b0[i][j][0] = b0[i][j][0] - ebl
    #             b1[i][j][0] = b1[i][j][0] + ebl
    #             out0 = self.feedforwardchecking(x, y, b0, self.weights)
    #             out1 = self.feedforwardchecking(x, y, b1, self.weights)
    #             nabla_b[i][j][0] = (out1 - out0) / (2*ebl)
    #             b0[i][j][0] = self.biases[i][j][0]
    #             b1[i][j][0] = self.biases[i][j][0]
    #
    #     w0 = deepcopy(self.weights)
    #     w1 = deepcopy(self.weights)
    #
    #     for i in xrange(0, len(self.weights)):
    #         for j in xrange(0, len(self.weights[i])):
    #             for k in xrange(0, len(self.weights[i][j])):
    #                 w0[i][j][k] = w0[i][j][k] - ebl
    #                 w1[i][j][k] = w1[i][j][k] + ebl
    #                 out0 = self.feedforwardchecking(x, y, self.biases, w0)
    #                 out1 = self.feedforwardchecking(x, y, self.biases, w1)
    #                 nabla_w[i][j][k] = (out1 - out0) / (2 * ebl)
    #                 w0[i][j][k] = self.weights[i][j][k]
    #                 w1[i][j][k] = self.weights[i][j][k]
    #     return (nabla_b, nabla_w)

    def unroll(self, bs, ws):
        flat = []
        for b, w in zip(bs, ws):
            flat.extend(b.ravel())
            flat.extend(w.ravel())
        return flat


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def pickle_param(bias, weights):
    pickle.dump(bias, open("../parameter/bias.p", "wb"))
    pickle.dump(weights, open("../parameter/weights.p", "wb"))
    return


def load_param():
    bias = pickle.load(open("../parameter/bias.p", "rb"))
    weights = pickle.load(open("../parameter/weights.p", "rb"))
    return bias, weights


