import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model
import copy

"""
A model for gaussian mixture models with dirichlet priors for mixture coeffecients

C ~ Dir(scale, shape)
U_i ~ N(0, 100), multivariate isotropic
S_i ~ |N(0, 10)|, half normal
Y ~ Multinomial(N, C)
X_n ~ N(U_{y_n}, S_{y_n}), multivariate

"""

np.random.seed(142857)
ndims = 1
nclusters = 3
nsamples = 2000
dirichlet_shape = 1./nclusters * np.ones([nclusters])
dirichlet_scale = 25
sd_halfnormal = 5.0
sd_epsilon = 0.01
mean_prior_mean = 0
mean_prior_sd = 100

# Generate data
C = np.random.dirichlet(dirichlet_shape * dirichlet_scale)
U = np.random.normal(np.zeros([ndims]), mean_prior_sd * np.eye(ndims), nclusters)
S = np.absolute(np.random.normal(0., sd_halfnormal, nclusters))
Y = np.random.multinomial(nsamples, C)
X = []
for i in xrange(nclusters):
    X.append(np.random.normal(U[i], S[i] * np.eye(ndims), Y[i]))
X_obs = np.concatenate((X[0], X[1], X[2]), 0)

#print X_obs
print C
print U
plt.plot(X_obs[:], np.ones(X_obs.shape), 'o', markersize=8)
plt.show()

# Infer class labels
from pymc3 import Dirichlet, Normal, MvNormal, HalfNormal, Categorical
import theano.tensor

with Model() as gmm:
    C = Dirichlet('mixture_coeff', dirichlet_scale * dirichlet_shape, shape = nclusters)
    S = HalfNormal('S', sd=sd_halfnormal, shape=nclusters)
    U = Normal('mu', mu=mean_prior_mean, sd=mean_prior_sd, shape=nclusters)
    Y = Categorical('labels', p=C, shape=nsamples)
    X = Normal('X', mu=U[Y], sd=S[Y], observed=X_obs)

from pymc3 import find_MAP
map_estimate = find_MAP(model=gmm)
print map_estimate


from pymc3 import NUTS, sample, Slice, Metropolis, ElemwiseCategorical, HamiltonianMC

modified_map_estimate = copy.deepcopy(map_estimate)
modified_map_estimate['mu'] = [1 if x < 0.001 else x for x in modified_map_estimate['mu']]

with gmm:
    # step = Slice(vars=[Y])
    # step = Metropolis(var=)
    start = copy.deepcopy(map_estimate)
    step1 = ElemwiseCategorical(vars=[Y])
    step2 = Metropolis(vars=[S, C, U])
    # trace = sample(100, step=step, start=map_estimate)
    trace = sample(20000, step=[step1, step2], start=start)

from pymc3 import traceplot
traceplot(trace)
plt.show()