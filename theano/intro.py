import numpy as np
import theano.tensor as T
from theano import function, pp
import theano

'''
Script where I take baby steps into Theano
TODO: convert to notebook?
'''

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
print 'Pretty print of z', pp(z)
f = function([x,y],z)
print 'Result of 1+2:', f(1,2)

# With matrixes, we may have to give in tag.test_value if we want debugging to be done on the fly
# Try commenting out the ().tag.test_value assignments!
X = T.dmatrix('X')
X.tag.test_value = np.random.rand(5,4)
Y = T.dmatrix('Y')
Y.tag.test_value = np.random.rand(5,4)
Z = X + Y
F = function([X,Y], Z)
Xin = np.array([[1,2], [3,4]])
Yin = np.array([[4,3], [5,6]])
print(F(Xin, Yin))

# By diabling this, we sidestep the issue...
theano.config.compute_test_value = "ignore"
X = T.dmatrix('X')
Y = T.dmatrix('Y')
Z = X + Y
F = function([X,Y], Z)
Xin = np.array([[1,2], [3,4]])
Yin = np.array([[4,3], [5,6]])
print F(Xin, Yin)

# Simple code for expansion of sums of equal-sized square matrices
A = T.dmatrix('A') 
B = T.dmatrix('B')
# C = np.linalg.matrix_power((A + B), 2) # <--- doesn't work here! numpy array operators only allow for array-like objects
C = T.dot(A, B)
print 'Pretty print of C:', pp(C)
D = function([A,B], C)
Ain = np.array([[1,2], [3,4]])
Bin = np.array([[4,3], [5,6]])
print 'Comparing numpy vs theano, note different in datatype'
print 'Theano:\n', D(Ain, Bin)
print 'numpy:\n', np.dot(Ain, Bin)