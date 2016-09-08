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
f = T.dvector('f') # data (single vector)
D = T.dmatrix('D') # dictionary
proj = T.dot(D.T, f)
amax = T.argmax(proj)
dist = proj[amax]
residual = f - T.squeeze(D[:, amax] * dist)
coeffs = T.zeros([D.shape[1], 1])
coeffs = T.set_subtensor(coeffs[amax], dist)

# single block
single_match = function([f, D], [coeffs, residual]) 
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
# ax3.setp(markerline, 'markerfacecolor', 'b')
ax3.set_ylim ([-1, 1])
ax4 = plt.subplot(224)
markerline, stemlines, baseline = ax4.stem(range(all_basis.shape[1]), c, '-.')
# ax4.setp(markerline, 'markerfacecolor', 'r', "marker", 'x')
ax4.set_ylim([-1, 1])
plt.show()