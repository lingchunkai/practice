import theano
import theano import function, shared
import theano.tensor as T
import numpy as np

'''
MLP
===
This is just to reinforce understanding of underlying gradient flow, not to replace standardized theano modules!

NOTE: all 32 bit in case we want to CUDA things in future!
'''

class FC_layer:
    def __init__(self, vec_in, n_in, n_out, activation='tanh'):
        self.vec_in = vec_in
        self.n_out = n_out
        self.n_in = n_in
        self.activation = activation

        # initial weights are selected from N(0, 1)
        initial_weights = np.random.normal(0., 1., [n_out, n_in])
        self.weights = shared(initial_weights)
        self.bias = shared(np.zeros([n_out]))

        tform = T.dot(self.weights, self.vec_in)
        # output = theano.scan(lambda v: T.tanh(tform) + self.bias, sequences=

def ConnectLayer(inputs, outputs):
    pass
