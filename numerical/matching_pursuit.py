import numpy as np
import theano
import theano.tensor as T
from theano import scan, function
import scipy
import matplotlib.pyplot as plt

SPARSITY_LEVEL = 10
DATA_LENGTH = 256

###############################################################
# Generate dictionary with maximum icoherance
fourier_basis = scipy.fftpack.idct(np.eye(DATA_LENGTH))
normalization_fourier_basis = np.sqrt(np.sum(fourier_basis ** 2, axis=1))
fourier_basis = fourier_basis / normalization_fourier_basis
spike_basis = np.eye(DATA_LENGTH)
all_basis = np.concatenate((spike_basis, fourier_basis.T), axis=1)

# Generate random, noisy sparse signal from said basis
choices = np.random.choice(all_basis.shape[1], SPARSITY_LEVEL, replace=False)
coeff_true = np.zeros([all_basis.shape[1], 1])
coeff_true[choices] = np.random.uniform(-1., 1., size=[SPARSITY_LEVEL, 1])
signal_true = np.dot(all_basis, coeff_true)
signal_noisy = signal_true + np.random.normal(0., 0.001, size=[signal_true.shape[0], 1])

###############################################################
# Matching pursuit (not OMP!)

theano.config.compute_test_value = "ignore"
class SingleMatchOp(object):
    def __init__(self, f, D):
        self.f = f
        self.D = D
        self.proj = T.dot(self.D.T, self.f)
        self.amax = T.argmax(T.abs_(self.proj))
        self.dist = self.proj[self.amax]
        self.residual = self.f - T.squeeze(self.D[:, self.amax] * self.dist)
        self.coeffs = T.zeros([self.D.shape[1], 1])
        self.coeffs = T.set_subtensor(self.coeffs[self.amax], self.dist)

class MultiMatchOp(object):
    def __init__(self, f, D, nSparsity):
        # assert(nSparsity > 0)
        self.X = [SingleMatchOp(f, D)]
        for n in xrange(1, nSparsity):
            self.X.append(SingleMatchOp(self.X[-1].residual, D))
        self.cumC = self.X[0].coeffs
        for n in xrange(1, nSparsity):
            self.cumC = self.cumC + self.X[n].coeffs

        self.op = function([self.X[0].f, self.X[0].D], [self.cumC, self.X[-1].residual])

    def __call__(self, a, b):
        return self.op(a, b)

print 'OMP, sparsity: ', SPARSITY_LEVEL
f1 = T.dvector('f1') # data (single vector)
D1 = T.dmatrix('D1') # dictionary
single_match = MultiMatchOp(f1, D1, 10)
# single block
c, res = single_match(np.squeeze(signal_noisy), all_basis)

plt.figure()
ax1 = plt.subplot(221)
ax1.plot(signal_noisy)
ax1.set_ylim([-1, 1])
ax2 = plt.subplot(223)
ax2.plot(res)
ax2.set_ylim([-1, 1])

ax3 = plt.subplot(222)
markerline, stemlines, baseline = ax3.stem(range(all_basis.shape[1]), coeff_true, '-.')
ax3.set_ylim ([-1, 1])
ax4 = plt.subplot(224)
markerline, stemlines, baseline = ax4.stem(range(all_basis.shape[1]), c, '-.')
ax4.set_ylim([-1, 1])
plt.show()
