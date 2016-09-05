import math
import numpy as np
import scipy.spatial.distance

# Ref: Achlioptas (2001)
# Ref: http://cseweb.ucsd.edu/~akmenon/HonoursThesis.pdf
# data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
# response = urllib2.urlopen(data_link)
# data = response.read()

MAX_DATA_SIZE = 100

def RandomProjection(A, k):
    '''
    @param A n * d data matrix
    @param ndims: dimensions for low dimensional
    '''
    n, d = data.shape
    R = np.random.choice([1,0,-1], size=(d,k), p=[1.0/6,2.0/3,1.0/6])
    R = R * math.sqrt(3)

    return np.dot(A, R) / math.sqrt(k)

with open('gisette_train.data.txt', 'r') as f:
    data_raw = f.readlines()

data = []
for d in data_raw:
    data.append(map(int, d.split()))
    if len(data) > MAX_DATA_SIZE: break
data = np.array(data)
print data.shape


orig_distances = scipy.spatial.distance.pdist(data)
orig_distances = scipy.spatial.distance.squareform(orig_distances)

for projdim in [10,50, 100, 500, 1000, 2500, 5000]:
    X = RandomProjection(data, projdim)
    projected_distances = scipy.spatial.distance.pdist(X)
    projected_distances = scipy.spatial.distance.squareform(projected_distances)

    # print orig_distances.shape, projected_distances.shape
    print 'With dimension', projdim, 'Frobenius norm is:', np.linalg.norm(orig_distances-projected_distances)