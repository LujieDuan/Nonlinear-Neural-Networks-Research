import time
import numpy as np


def bench_matrix():
    print "Testing matrix multiplication:"
    start = time.time()
    matrix = np.random.randn(1000, 1000)
    for i in range(0, 1000):
        m = matrix * matrix
    print time.time() - start


print "Benchmark Started!"
bench_matrix()
