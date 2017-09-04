# Research on Neural Networks with Nonlinear Synapses

##  Part 1:

Benchmark on two linear algebra libraries setup: Numpy with Python, and Breeze with Scala. 

The operations that we used the most in a neural networks are matrix multiplication, matrix apply (apply operation elementwise), and matrix dot multiplication (elementwise multiplication). 



##  Part 2: 

A Python/Numpy implementation of ordinary linear neural networks, and a corresponding sequential implementation in Scala/Breeze; both trained on MNIST handwritten digit database. 


##	Part 3:

A Scala/Breeze implementation of neural networks with nonlinear synapses, trained on MNIST handwritten digit database. 


##	Part 4:

Using gradient checking to check the correctness of neural networks with nonlinear sysnapses. 


##	Part 5:

A Scala/Breeze implementation of a variation of neural networks with nonliear sysapses - each transformation function (synapse) now only keep one exponent term. We call this as neural networks with single term nonlinear synapse. 


##	Part 6:

Test two kinds of nonlinear networks on XOR classification - a classical problem, that for ordinary neural networks with sigmoid activation function, they need minimum 5 noods in 3 layers, including the input layer. 


##	Part 7: 

Test ordinary neural networks and two nonlinear kinds on [IPPN arabidopsis leaf counting data set](https://www.plant-phenotyping.org/datasets-home).


##	Part 8: 

Test with [One-hundred plant species leaves data set](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set). 


##	Part 9: 

Test neural networks with a mix of linear/nonlinear synapses.


##	Part 10:

Test nonlinear network with ReLU, Gaussian activation functions and compare with Sigmoid funtion. 


##	Part 11:

Change the data type of floating point number from double precision(Double) to single precision(Float). 


##	Part 12: 

Data parallelism: using parameter server method and Akka actor to train nonlinear networks in parallel. 


##	Part 13: 

Model parallelism: using Akka actor to devide one model of nonlinear netowrks and train in parallel. 


##	Part 14:

Implement DistBelief with Akka actor: a combination of data parallelism and model parallelism. 


##	Part 15: 

Experiments translated from the original paper. 


##	Part 16: 

Using different learning ratio for coefficients and exponents on neural networks with nonlinear synapses. 


##	Part 16: 

An analysis of native GPU operations. 
