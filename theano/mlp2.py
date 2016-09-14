"""
This implementation of an MLP follows closely from  the theano official tutorials:
[1] : http://deeplearning.net/tutorial/mlp.html
[2] : http://deeplearning.net/tutorial/logreg.html

#1. It is close to our first implementation but deals with the nitty gritty details in Theano.
#2. Warning: inputs are given in a n x d matrix, where:
    n is the size of test/train minibatch,
    d is the dimension of input

    This implies that the conveiton of direction of dot products are in the 
    opposite direction (transpose), rather than the usual M * col format.
"""

import os
import sys
import timeit
import pickle
import gzip

import numpy as np

import theano
import theano.tensor as T

theano.config.compute_test_value = "off"
theano.config.exception_verbosity = 'high'

class LogisticRegression(object):
    """
    Module to perform linear logistic classification
    """

    def __init__(self, inputs, n_in, n_out):
        '''
        @param n_in: fan-in
        @param n_out: fan-out
        '''

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out, ),
                dtype=theano.config.floatX
                ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        # ^ should be equivalent to:
        #   self.exps = T.exp(T.dot(self.W, inputs) + self.b)
        #   self.p_y_given_x = self.exps / T.sum(self.exps)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Book-keeping
        self.inputs = inputs
        self.n_in = n_in
        self.n_out = n_out
        self.params = [self.W, self.b]

    def NegativeLogLikelihood(self, y):
        '''
        Computes the negative log-likelihood as a *mean* rather than a sum 
        to be invariant to minibatch size

        Likelihood is computed w.r.t. the true class, i.e. p(y_pred == y|x)

        @param y: theano.tensor.TensorType (row) containing the target labels
        '''

        return T.mean(-T.log(self.p_y_given_x[T.arange(y.shape[0]), y]))

    def Errors(self, y):
        '''
        Compute the percentage of errors over the minibatch
        '''

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should be of the same shape as y_pred',
                ('y', y.ndim, 'y_pred', self.y_pred.ndim)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError(
                'implementation only supports integer labels'
                )


class HiddenLayer(object):
    def __init__(self, inputs, n_in, n_out, W=None, b=None, activation=T.tanh, rng=np.random.RandomState(1234)):
        '''
        @param inputs: theano.tensor.TensorType, inputs
        @param n_in: fan-in
        @param n_ut: fan-out
        @param W, b: theano.tensor.TensorType, initial weights and bias 
        [defaults to standard config for tanh activation]
        @param activation: theano function for activation
        [defaults to tanh, reverts to linear (identity activation) if set to None]
        '''

        # Default initialization for W
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                    ),
            dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        # Default initialization for b
        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        lin_output = T.dot(inputs, W) + b
        self.outputs = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # book-keeping
        self.inputs = inputs
        self.b = b
        self.W = W
        self.params = [self.W, self.b]

    @staticmethod
    def Test():
        x = T.matrix(name='x')
        y = T.ivector('y')
        hl = HiddenLayer(x, 3, 5)


class MLP(object):
    def __init__(self, inputs, n_in, n_out, n_layer_sizes, layer_types=None, rng=np.random.RandomState(1234)):
        '''
        @param inputs: theano.tensor.TensorType, inputs
        @param n_layer_sizes: [...] array of positive integers containing size of each hidden layer
        @param layer_types: [...] array of object types
        '''

        # By default, we use fully-connected hidden layers
        if layer_types is None:
            layer_types = [HiddenLayer for n in xrange(len(n_layer_sizes))]

        if len(n_layer_sizes) != len(layer_types):
            raise TypeError(
                'n_layer_sizes and number of layer_types do not match',
                ('n_layer_sizes', n_layer_sizes, 'layer_types', layer_types)
            )

        hidden_layers = []
        prev_outputs = inputs
        n_prev_outputs = n_in
        for n_layer in xrange(len(n_layer_sizes)):
            layer = layer_types[n_layer](
                inputs=prev_outputs,
                n_in=n_prev_outputs,
                n_out=n_layer_sizes[n_layer],
                rng=rng
            )  # TODO: add optional kwargs

            prev_outputs = layer.outputs
            n_prev_outputs = n_layer_sizes[n_layer]
            hidden_layers.append(layer)

        output_layer = LogisticRegression(prev_outputs, n_prev_outputs, n_out)

        self.L1 = (
            sum([abs(x.W).sum() for x in hidden_layers]) +
            abs(output_layer.W).sum()
        )

        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in hidden_layers]) +
            (output_layer.W ** 2).sum()
        )

        # book-keeping
        self.inputs = inputs
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.params = [p for sublayer in hidden_layers for p in sublayer.params] + output_layer.params

    @staticmethod
    def Test():
        x = T.matrix(name='x')
        y = T.ivector('y')
        mlp = MLP(inputs=x, n_in=5, n_out=3, n_layer_sizes=[2, 3, 4])


def LoadData(dataset):
    '''
    @param dataset: path to dataset
    '''

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        folders_to_check = ['data', '.']
        for f in folders_to_check:
            # Check if dataset is in the data folder
            new_path = os.path.join(
                os.path.split(__file__)[0],
                f,
                dataset
            )

            if os.path.isfile(new_path):
                dataset = new_path
                break

    print('loading data...')
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    # Format for pickle input.
    # Note: copied from [2]
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def SharedDataset(data_xy, borrow=True):
        '''
        Load dataset into shared variables which allows for fast GPU implementations

        @param: data_xy, tuple of data-labels
        '''

        data_x, data_y = data_xy

        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow
        )

        shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX),
            borrow=borrow
        )

        # current implementation focuses on classification, hence integer labels
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = SharedDataset(train_set)
    test_set_x, test_set_y = SharedDataset(test_set)
    valid_set_x, valid_set_y = SharedDataset(valid_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


def TrainMLP(learning_rate=0.01, L1_reg=0., L2_reg=0.0001, n_epochs=1000, batch_size=20, n_layer_sizes=[500], dataset='mnist.pkl.gz'):
    '''
    '''

    ####################
    # Extract datasets #
    ####################
    datasets = LoadData(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute size of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('building the model')

    ##################
    # Model building #
    ##################

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    # construct MLP class
    mlp = MLP(
        inputs=x,
        n_in=28*28,
        n_out=10,
        n_layer_sizes=n_layer_sizes,
        rng=rng
    )

    cost = (
        mlp.output_layer.NegativeLogLikelihood(y) +
        L1_reg * mlp.L1 +
        L2_reg * mlp.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[mlp.output_layer.Errors(y)],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[mlp.output_layer.Errors(y)],
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in mlp.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(mlp.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=[mlp.output_layer.Errors(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    ###############
    # Train Model #
    ###############
    print('Training...')

    patience = 10000  # minimum examples to look at
    patience_inc = 2  # increase in looks when new hi is reached
    improvement_threshold = 0.995  # relative improvement to be considered significant
    validation_frequency = min(n_train_batches, patience // 2)  # go through this many minibatches before checking validation set

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            n_iter = (epoch - 1) * n_train_batches + minibatch_index

            if (n_iter + 1) % validation_frequency == 0:
                # Compute 0-1 validation loss
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation_error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # Update best validation score if needed
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        # increase patience count
                        patience = max(patience, n_iter * patience_inc)

                    best_validation_loss = this_validation_loss
                    best_iter = n_iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        '     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                        (epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.)
                    )

            if patience <= n_iter:
                done_looping = True
                break  # break out of outermost while loop

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.))

TrainMLP(learning_rate=0.01)
