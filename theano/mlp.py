import theano
from theano import function, shared, pp
import theano.tensor as T
import numpy as np

'''
MLP
===
This is just to reinforce understanding of underlying gradient flow, not to replace standardized theano modules!

NOTE: all 32 bit in case we want to CUDA things in future!
'''
theano.config.compute_test_value = "ignore"
theano.config.exception_verbosity = 'high'


class FC_layer:
    def __init__(self, vec_in, n_in, n_out, activation='tanh'):
        self.vec_in = vec_in
        self.n_out = n_out
        self.n_in = n_in
        self.activation = activation

        # initial weights are selected from N(0, 1)
        initial_weights = np.random.normal(0., 1., [n_out, n_in])
        self.weights = shared(initial_weights)
        self.bias = shared(np.zeros([n_out, 1]))

        tmp = T.dot(self.weights, self.vec_in) + self.bias
        self.output = T.tanh(tmp)


class MLP:
    def __init__(self, layer_sizes):
        self.inputs = T.dmatrix('mlp_inputs')
        self.output_nodes = [self.inputs]
        self.layers = []  # excludes first layer
        for x in xrange(1, len(layer_sizes)):
            self.layers.append(FC_layer(self.output_nodes[-1], layer_sizes[x-1], layer_sizes[x]))
            self.output_nodes.append(self.layers[-1].output)
        self.outputs = self.layers[-1].output
        self.func = function([self.inputs], [self.outputs])

    def __call__(self, inputs):
        return self.func(inputs)

mlp = MLP([7,3,6,2])
print pp(mlp.outputs)
print mlp(np.array([[1],[2],[3],[4],[5],[6],[7]]))
