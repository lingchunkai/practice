import theano
import theano.tensor as T
from theano import function, shared, In
import numpy as np
import copy

theano.config.compute_test_value = "ignore"

# Average filtering a stream of data
N_AVG = 5
hist = [shared(0.) for x in xrange(N_AVG)]

in_num = T.dscalar('in_num')
out_avg = T.sum(hist) / 2.0
averager = function([in_num], out_avg, updates=[(hist[x+1], hist[x]) for x in xrange(N_AVG-1)]  + [(hist[0], in_num)])

print averager(1) # Prints sum that was _before_ the new update, will have to design differently for output _after_ update
print averager(1)
print averager(1)
print averager(1)
print averager(1)
print averager(1)

####################################################################################
# MDP solver via value iteration

reward_function = np.array([[0, 0, 0, 0, 100], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
# transition is 60 in direction chosen, 20 in each right angle, stay in location if bump into wall
discount = 0.95
HEIGHT = len(reward_function)
WIDTH = len(reward_function[0])
value_estimate = np.zeros(reward_function.shape)

in_value_estimate = T.dmatrix('in_value_estimate')
in_value_estimate.tag.test_value = np.random.rand(HEIGHT, WIDTH)
future_reward = [None] * 4
future_reward[0] = T.concatenate((in_value_estimate[0:1, 0:WIDTH], in_value_estimate[0:(HEIGHT-1), 0:WIDTH]), axis=0)
future_reward[1] = T.concatenate((in_value_estimate[1:HEIGHT], in_value_estimate[HEIGHT-1:HEIGHT, :]), axis=0)
future_reward[2] = T.concatenate((in_value_estimate[:, 0:1], in_value_estimate[:, 0:WIDTH-1]), axis=1)
future_reward[3] = T.concatenate((in_value_estimate[:, 1:WIDTH], in_value_estimate[:, WIDTH-1:WIDTH]), axis=1)

future_reward_full = [None] * 4
future_reward_full[0] = future_reward[0] * 0.5 + future_reward[2] * 0.25 + future_reward[3] * 0.25
future_reward_full[1] = future_reward[1] * 0.5 + future_reward[2] * 0.25 + future_reward[3] * 0.25
future_reward_full[2] = future_reward[2] * 0.5 + future_reward[0] * 0.25 + future_reward[1] * 0.25
future_reward_full[3] = future_reward[3] * 0.5 + future_reward[0] * 0.25 + future_reward[1] * 0.25
future_reward_full[0] = future_reward_full[0].dimshuffle((0,1,'x'))
future_reward_full[1] = future_reward_full[1].dimshuffle((0,1,'x'))
future_reward_full[2] = future_reward_full[2].dimshuffle((0,1,'x'))
future_reward_full[3] = future_reward_full[3].dimshuffle((0,1,'x'))
future_reward_full = T.concatenate(future_reward_full, axis=2)
future_reward_full = T.max(future_reward_full, axis=2)
out_value_estimate = reward_function + discount * future_reward_full
BellmanIteration = function([in_value_estimate], out_value_estimate)

for iters in xrange(300):
    old_value_estimate = copy.deepcopy(value_estimate)
    value_estimate = BellmanIteration(value_estimate)
    print 'Bellman residual: ', np.linalg.norm(value_estimate - old_value_estimate)

print 'Value function: '
print value_estimate.astype(int)