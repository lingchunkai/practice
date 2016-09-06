import numpy as np
from scipy import stats
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP

# generate data
np.random.seed(142857)
sigma = 1.
alpha, beta = 1., [1, .25]

n_samples = 1000
X = [np.asfarray(range(n_samples)), np.asfarray(range(n_samples))/5.0]
Y = alpha + X[0] * beta[0] + X[1] * beta[1]

lin_regress = Model()
with lin_regress:

    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)

    # Noiseless, deterministic (given alpha and beta) observation
    mu = alpha + X[0] * beta[0] + X[1] * beta[1]

    # Create observed random variable
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

map_estimate = find_MAP(model=lin_regress)
print map_estimate

