import numpy as np
import time 

def timing(f):
    '''
    Decorator to print timing for function
    Source: http://stackoverflow.com/questions/5478351/python-time-measure-function
    '''
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

@timing
def GenerateGramMatrix(v):
    '''
    Rows of orthogonal vectors
    '''
    weights = np.random.uniform(0., 1., v.shape[0])
    v = np.dot(np.diag(weights), v)
    return np.dot(v, v.T)

@timing
def GenerateBasis(ndims):
    v = np.random.multivariate_normal(np.zeros(ndims), np.eye(ndims, ndims), ndims)
    b, _ = np.linalg.qr(v)
    return b

@timing
def Nystrom(M, k, l):
    '''
    Nystrom uniform sampling of columns. 
    Note that other sampling (eg. weighted by norm of columns or their diagonals) also exist
    '''
    ndims = M.shape[0]
    samples = np.random.choice(ndims, l, replace=False) # sample k columns uniformly
    perm = samples.tolist() + [x for x in xrange(ndims) if x not in samples]
    Mt = M[np.ix_(perm, perm)]

    W = Mt[np.ix_(range(l), range(l))]
    C = Mt[np.ix_(range(ndims), range(l))]
    U, D, V = np.linalg.svd(W)

    # Compute rank k pseudoinverse of W
    D = [1/D[x] if x >= k else 0 for x in xrange(l)]
    Wpinv = np.dot(np.dot(U, np.diag(D)), V)
    X = np.dot(np.dot(C, Wpinv), C.T)

    # Unpermute
    iperm = np.zeros(ndims, dtype=int)
    iperm[perm] = range(ndims)

    return X[np.ix_(iperm, iperm)]

@timing
def Optimal(M, k):
    ndims = M.shape[0]
    U, D, V = np.linalg.svd(M)
    D[range(k, ndims)] = 0
    return np.dot(np.dot(U, np.diag(D)), V)

n = 1000
b = GenerateBasis(n)
gram = GenerateGramMatrix(b)


k = 100
frob_optimal = np.linalg.norm(gram-Optimal(gram, k))
print 'Frob error optimal', frob_optimal

for l in [100, 200, 300, 400]:
    N = Nystrom(gram, k, l)
    frob = np.linalg.norm(gram-N)
    print 'Frob error approximation', frob, 'l =', l, 'Ratio: ', frob_optimal/frob
