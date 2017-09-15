import time
import numpy as np


def bench_matrix_elementwise_multiplication():
    print "Testing matrix elementwise multiplication:"
    start = time.time()
    matrix = np.random.randn(1000, 1000)
    for i in range(0, 1000):
        m = matrix * matrix
    print "Total: {0}s".format(time.time() - start)


def bench_matrix_apply():
    print "Testing matrix apply:"
    start = time.time()
    matrix = np.random.randn(1000, 1000)
    for i in range(0, 1000):
        m = np.exp(matrix)
    print "Total: {0}s".format(time.time() - start)


def bench_matrix_multiplication():
    print "Testing matrix multiplication:"
    start = time.time()
    matrix = np.random.randn(1000, 1000)
    for i in range(0, 1000):
        m = np.dot(matrix, matrix)
    print "Total: {0}s".format(time.time() - start)


print "Benchmark Started!"
bench_matrix_elementwise_multiplication()
bench_matrix_apply()
bench_matrix_multiplication()
