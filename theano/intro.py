import numpy as np
import theano.tensor as T
from theano import function, pp
import theano

'''
Script where I take baby steps into Theano
TODO: convert to notebook?
'''

#############################################################
# Part I

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


#############################################################
# Part II

# Logistic regression on product of matrices (depends on previous part)
E = 1.0/(1.0+T.exp(-C)) # <--- Note broadcasting rules apply here
print pp(E)
fE = function([A,B], E)
# Lets define Ain, Bin to be transition matrices! (rows must sum to 1 - we are using the statisticians convention where p' = pM)
Ain = np.array([[0.2, 0.8], [0.4, 0.6]])
Bin = np.array([[0.8, 0.2], [0.5, 0.5]])
print fE(Ain, Bin) # Qn: is there a way we can extract the intermediate computed result, or do theano optimizations obscure these from us?

# Multiple outputs: lets compute the different norms between vectors x and y!
x = T.dvector('x')
y = T.dvector('y')
L1 = T.sum(T.abs_(x - y))
L2 = T.sqrt(T.sum((x - y) ** 2))
Linf = T.max(T.abs_(x-y))
fNorm = function([x,y], [L1, L2, Linf])
xin = np.array([7,11])
yin = np.array([8,31])
print 'Theano norms:', fNorm(xin, yin)
print 'Numpy norms:', np.linalg.norm(xin-yin, 1), np.linalg.norm(xin-yin, 2), np.linalg.norm(xin-yin, float('inf'))
