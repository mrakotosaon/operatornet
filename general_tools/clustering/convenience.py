'''
Created on Aug 31, 2017

@author: optas
'''
import warnings

try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn library is not installed.')


def find_nearest_neighbors(X, Y=None, k=10):
    ''' find_nearest_neighbors.

    If Y is not provided, it returns the k neighbors of each data point in the X dataset.
    Otherwise, it returns the k neighbors of X in Y.

    Input: X, Y: [n_samples, n_features]
    '''
    s = 0
    if Y is None:
        Y = X
        k = k + 1   # First neighbor is one's shelf.
        s = 1       # Used to drop the first-returned neighbor if needed.

    nn = NearestNeighbors(n_neighbors=k).fit(Y)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, s:]
    indices = indices[:, s:]
    return indices, distances
