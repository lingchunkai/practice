import theano
from theano import function, shared, pp
import theano.tensor as T
import theano.tensor.nlinalg
import numpy as np
import matplotlib.pyplot as plt

'''
MLP
===
This is just to reinforce understanding of underlying gradient flow, not to replace standardized theano modules!
This is obviously very poorly written (eg. uses fixed learning rate, single point SGD etc);     

NOTE: all 32 bit in case we want to CUDA things in future!
'''
theano.config.compute_test_value = "warn"
theano.config.exception_verbosity = 'high'


class FC_layer:
    def __init__(self, vec_in, n_in, n_out, activation='relu'):
        self.vec_in = vec_in
        self.n_out = n_out
        self.n_in = n_in
        self.activation = activation

        # initial weights are selected from N(0, 1)
        initial_weights = np.random.normal(0., 1., [n_out, n_in])
        self.weights = shared(initial_weights)
        self.bias = shared(np.zeros([n_out, 1]))

        tmp = T.dot(self.weights, self.vec_in)
        if activation == 'tanh':
            self.output = T.tanh(tmp) + self.bias
        elif activation == 'relu':
            self.output = T.nnet.relu(tmp) + self.bias


class MLP:
    def __init__(self, layer_sizes, learning_rate = 0.1, regularization = 0.05, activation='relu'):

        # Forward pass
        self.activation = activation
        self.inputs = T.dmatrix('mlp_inputs')
        self.inputs.tag.test_value = np.random.normal(0., 1., [layer_sizes[0], 1])
        self.output_nodes = [self.inputs]
        self.layers = []  # excludes first layer
        for x in xrange(1, len(layer_sizes)):
            self.layers.append(FC_layer(self.output_nodes[-1], layer_sizes[x-1], layer_sizes[x], activation))
            self.output_nodes.append(self.layers[-1].output)
        self.outputs = self.layers[-1].output

        # Backward pass
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.groundtruth = T.dmatrix('groundtruth')
        self.groundtruth.tag.test_value = np.random.normal(0., 1., [layer_sizes[-1], 1])
        self.error = T.sum((self.groundtruth-self.outputs) ** 2)
        self.loss = self.error + self.regularization * T.sum(T.concatenate([T.flatten(x.weights, outdim=1) for x in self.layers], axis=0) ** 2)
        self.gradients = [T.grad(self.loss, self.layers[x].weights) for x in xrange(len(self.layers))]
        self.gradients_bias = [T.grad(self.loss, self.layers[x].bias) for x in xrange(len(self.layers))]

        self.backprop = function([self.inputs, self.groundtruth], self.gradients + [self.error, self.loss], updates= 
            [(self.layers[x].weights, self.layers[x].weights - self.learning_rate * self.gradients[x]) for x in xrange(len(self.layers))] + 
            [(self.layers[x].bias, self.layers[x].bias - self.learning_rate * self.gradients_bias[x]) for x in xrange(len(self.layers))])
        self.forward = function([self.inputs], [self.outputs])

    def __call__(self, inputs):
        return self.forward(inputs)

    '''
    def backprop(self, data_in, output, learning_rate):
        Single backpropagation step
    '''

print 'Sanity check, learn a point function'
mlp = MLP([7, 3, 6, 2], learning_rate=0.05)
print pp(mlp.outputs)
for k in xrange(100):
    res = mlp.backprop(np.array([[1], [2], [3], [4], [5], [6], [7]]), np.array([[5], [3]]))
    print 'loss: ', res[-1], 'error: ', res[-2]
print 'Final prediction: ', mlp(np.array([[1], [2], [3], [4], [5], [6], [7]]))

# Try to learn quadratic function
INPUT_DIMENSION = 2
N_TRAIN = 1000
N_VALIDATION = 1000
A = np.random.normal(0., 1., [INPUT_DIMENSION, INPUT_DIMENSION])
quadratic_part = lambda x: np.dot(np.dot(x.T, A), x)
linear_part = lambda x: np.array([[1+4 * x], [3-2*x]])
func_to_learn = lambda x: linear_part(quadratic_part(x))

train_in = np.random.uniform(-1., 1., [INPUT_DIMENSION, N_TRAIN])
train_out = [func_to_learn(train_in[:, n]) for n in xrange(N_TRAIN)]
validation_in = np.random.uniform(-1., 1., [INPUT_DIMENSION, N_VALIDATION])
validation_out = [func_to_learn(train_in[:, n]) for n in xrange(N_TRAIN)]

# Stochastic gradient descent
BATCH_SIZE = 1
N_BATCHES = 250
mlp = MLP([INPUT_DIMENSION, 10, 10, 10, 10, 10, 2], learning_rate=0.0005, regularization=0.05)

histloss = [None] * N_BATCHES
plt.ion()
fig = plt.figure()
hl, = plt.plot(range(N_BATCHES), histloss)
plt.show()
plt.title('Loss')
plt.pause(0.1)
for m in xrange(N_BATCHES):
    sample_order = np.random.permutation(N_TRAIN)
    terror, tloss = 0, 0
    for n in sample_order:
        gradsloss = mlp.backprop(np.reshape(train_in[:, n], [INPUT_DIMENSION, 1]), train_out[n])
        tloss += gradsloss[-1]
        terror += gradsloss[-2]
    histloss[m] = tloss
        
    hl.set_ydata(histloss)
    plt.gca().relim()
    plt.gca().autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.1)
        
    print 'Training loss: ', tloss, 'Training error: ', terror

out = [np.linalg.norm(mlp(np.reshape(validation_in[:, n], [INPUT_DIMENSION, 1])) - validation_out[n])**2 for n in xrange(N_VALIDATION)]
print 'Validation loss: ', sum(out)
