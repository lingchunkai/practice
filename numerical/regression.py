import numpy as np
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt
import math

'''
Regression
==========
Explore some of the properties of common regressors in the underdetermined setting

Notice how the lasso results in a mix of the 'scaling effect' from ridge regression and L0(hard thresholding)
'''

def OLS(signals, basis):
    return np.dot(covariates.T, signals)

def RidgeRegression(observations, lamb, covariates):
    '''
    Minimize {1/N ||y-Xb||^2 + lambda||b||^2}
    '''
    N = len(observations)
    ols_optimal = OLS(observations, covariates)
    return ols_optimal / (1 + N * lamb)

def LassoRegression(observations, lamb, covariates):
    N = len(observations)
    ols_optimal = OLS(observations, covariates)
    tmp = 1-(N*lamb)/np.abs(ols_optimal)
    t = np.concatenate([tmp, np.zeros(tmp.shape)], axis=1)
    t = np.reshape(np.max(t, axis=1), ols_optimal.shape)
    return t * ols_optimal

def l0Regression(observations, lamb, covariates):
    N = len(observations)
    ols_optimal = OLS(observations, covariates)
    tmp = np.where(np.abs(ols_optimal) > math.sqrt(lamb * N))
    ret = np.zeros(ols_optimal.shape)
    ret[tmp] = 1
    return ret * ols_optimal


# Generate time series
SIGNAL_LENGTH = 256
N_COVARIATES = 64
N_DATA = 2048
NOISE_LEVEL = 0.1
fourier_basis = scipy.fftpack.idct(np.eye(SIGNAL_LENGTH))
normalization_fourier_basis = np.sqrt(np.sum(fourier_basis ** 2, axis=1))
fourier_basis = fourier_basis / normalization_fourier_basis
choices = np.random.choice(fourier_basis.shape[1], N_COVARIATES, replace=False)
covariates = fourier_basis[choices, :]

soln = np.random.normal(0, 1.0, [SIGNAL_LENGTH, 1])
true_obs = np.dot(covariates, soln)
observations = true_obs + np.random.normal(0, NOISE_LEVEL, true_obs.shape)

for reg in [0.0001, 0.001, 0.005, 0.01, 0.05]:
    ridge_ans = RidgeRegression(observations, reg, covariates)
    lasso_ans = LassoRegression(observations, reg, covariates)
    l0_ans = l0Regression(observations, reg, covariates)

    plt.figure()
    plt.suptitle('lambda = ' + str(reg))
    ax1 = plt.subplot(423)
    ax1.plot(np.dot(covariates, ridge_ans))
    plt.title('Ridge, reconstructed')
    ax2 = plt.subplot(425)
    ax2.plot(np.dot(covariates, lasso_ans))
    plt.title('Lasso, reconstructed')
    ax3 = plt.subplot(427)
    ax3.plot(np.dot(covariates, l0_ans))
    plt.title('L0, reconstructed')

    ax4 = plt.subplot(424)
    markerline, stemlines, baseline = ax4.stem(range(SIGNAL_LENGTH), ridge_ans, '-.')
    plt.title('Ridge, coeffs')
    ax5 = plt.subplot(426)
    markerline, stemlines, baseline = ax5.stem(range(SIGNAL_LENGTH), lasso_ans, '-.')
    plt.title('Lasso, coeffs')
    ax6 = plt.subplot(428)
    markerline, stemlines, baseline = ax6.stem(range(SIGNAL_LENGTH), l0_ans, '-.')
    plt.title('L0, coeffs')

    axorig = plt.subplot(421)
    axorig.plot(true_obs)
    axorig_coeff = plt.subplot(422)
    markerline, stemlines, baseline = axorig_coeff.stem(range(SIGNAL_LENGTH), soln, '-.')

    plt.show()
