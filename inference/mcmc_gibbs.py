import numpy as np
import matplotlib.pyplot as plt
import copy

"""
Sample from bivariate normal distribution
"""

COV = np.array([[1, -0.5], [-0.5, 2]])
MEAN = np.array([1.0, 2.0])
N_SAMPLES = 5000

# Sample from true distribution
plt.figure(1)
x_true, y_true = np.random.multivariate_normal(MEAN, COV, N_SAMPLES).T
plt.plot(x_true, y_true, 'o')
plt.axis('equal')
plt.draw()

# Sample using gibbs sampling *NEVER DO THIS IRL!*
# start_point = MEAN # Begni
start_point = np.array([0.0, 0.0])
print 'Starting point: ', start_point
n_loops = N_SAMPLES / 2

cond_variance = []
regression_coeff = []
for index_to_update in xrange(2):
    # regression_coeff = copy.deepcopy(COV)
    tmp_cov = np.delete(COV, index_to_update, 0)
    tmp_cov = np.delete(tmp_cov, [x for x in xrange(2) if x != index_to_update], 1)
    tmp_var = np.delete(COV, [x for x in xrange(2) if x == index_to_update], 0)
    tmp_var = np.delete(tmp_var, [x for x in xrange(2) if x == index_to_update], 1)

    regression_coeff.append(np.linalg.solve(tmp_var.T, tmp_cov).T)
    # print 'Regression coeffs: ', regression_coeff[-1]
    cond_variance.append(COV[index_to_update, index_to_update] - np.dot(regression_coeff[-1], tmp_cov))
    # print 'Conditional variance: ', cond_variance[-1]

cur_point = copy.deepcopy(start_point)
gibbs_samples = np.zeros([N_SAMPLES, 2])
for i_loop in xrange(n_loops):
    for index_to_update in xrange(2):
        other_pos = cur_point[[x for x in xrange(2) if x != index_to_update]]
        other_mean = MEAN[[x for x in xrange(2) if x != index_to_update]]
        post_mean = MEAN[index_to_update] + np.dot(regression_coeff[index_to_update], other_pos - other_mean)
        cur_point[index_to_update] = np.random.multivariate_normal(post_mean, cond_variance[index_to_update])
        gibbs_samples[i_loop*2+index_to_update, :] = cur_point

print gibbs_samples.shape
plt.figure(2)
plt.plot(gibbs_samples[:, 0], gibbs_samples[:, 1], 'o')
plt.axis('equal')
plt.draw()

print 'Gibbs mean: ', np.mean(gibbs_samples, axis=0)
print 'Gibbs variance:', np.cov(gibbs_samples.T)
plt.show()