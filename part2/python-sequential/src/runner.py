"""
Main function to run the networks

Copyright @ Lujie Duan
"""


import mnist_loader
import network


training_data_vectorized, training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Run with networks of 784 - 30 - 10 on MNIST, as a classification task
net = network.Network([784, 30, 10])
net.SGD(training_data_vectorized, training_data, 30, 100, 3.0, validation_data, test_data)
