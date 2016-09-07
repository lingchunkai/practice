import numpy as np
import theano
import theano.tensor as T
from theano import scan, function
import scipy
import matplotlib.pyplot as plt

SPARSITY_LEVEL = 10
DATA_LENGTH = 256

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
plt.plot(signal_noisy)
plt.show()

###############################################################
# Matching pursuit.

f = T.dvector('f') # data (single vector)
D = T.dmatrix('D') # dictionary
proj = T.dot(D, f)
amax = T.argmax(proj)
dist = proj[amax]
residual = f - T.dot(D, D[amax, :]) * dist
coeffs = T.zeros([D.shape[1], 1])
T.set_subtensor(coeffs[amax], dist)
match_one = function([f, D], [coeffs, residual])


