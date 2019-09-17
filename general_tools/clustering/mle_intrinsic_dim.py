"""This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.
(c) 2011-2016, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>

This file is based on a Matlab script by Elizaveta Levina, University of 
Michigan, available at http://dept.stat.lsa.umich.edu/~elevina/mledim.m
Reference:  E. Levina and P.J. Bickel (2005).  
 "Maximum Likelihood Estimation  of Intrinsic Dimension."  
 In Advances in NIPS 17, Eds. L. K. Saul, Y. Weiss, L. Bottou. 
"""

import numpy as np


def intrinsic_dimension(X, k1=6, k2=12, estimator='levina', trafo='var', mem_threshold=5000):
    """Calculate intrinsic dimension based on the MLE by Levina and Bickel [1].

    Parameters
    ----------
    X : ndarray
        - An ``m x n`` vector data matrix with ``n`` objects in an 
          ``m`` dimensional feature space 
        - An ``n x n`` distance matrix.
        
    k1 : int, optional (default: 6)
        Start of neighborhood range to search in.
        
    k2 : int, optional (default: 12)
        End of neighborhood range to search in.
        
    estimator : {'levina', 'mackay'}, optional (default: 'levina')
        Determine the summation strategy: see [2].

    trafo : {None, 'std', 'var'}, optional (default: 'var')
        Transform vector data.

        - None: no transformation
        - 'std': standardization 
        - 'var': subtract mean, divide by variance (default behavior of 
          Laurens van der Maaten's DR toolbox; most likely for other 
          ID/DR techniques).
    mem_treshold : int, optional, default: 5000
        Controls speed-memory usage trade-off: If number of points is higher
        than the given value, don't calculate complete distance matrix at
        once (fast, high memory), but per row (slower, less memory).
    Returns
    -------
    d_mle : int
        Intrinsic dimension estimate (rounded to next integer)

    NOTE: the MLE was derived for euclidean distances. Using 
    other dissimilarity measures may lead to undefined results.

    References
    ----------
        [1] Levina, E., Bickel, P. (2004)
        Maximum Likelihood Estimation of Intrinsic Dimension
        https://www.stat.berkeley.edu/~bickel/mldim.pdf
        
        [2] http://www.inference.phy.cam.ac.uk/mackay/dimension
    """
    n = X.shape[0]
    if estimator not in ['levina', 'mackay']:
        raise ValueError("Parameter 'estimator' must be 'levina' or 'mackay'.")
    if k1 < 1 or k2 < k1 or k2 >= n:
        raise ValueError("Invalid neighborhood: Please make sure that 0 < k1 <= k2 < n. (Got k1={} and k2={}).".format(k1, k2))
    X = X.copy().astype(float)

    # New array with unique rows   (    % Remove duplicates from the dataset )
    X = X[np.lexsort(np.fliplr(X).T)]

    if trafo is None:
        pass
    elif trafo == 'var':
        X -= X.mean(axis=0)
        X /= X.var(axis=0) + 1e-7
    elif trafo == 'std':
        # Standardization
        X -= X.mean(axis=0)
        X /= X.std(axis=0) + 1e-7
    else:
        raise ValueError("Transformation must be None, 'std', or 'var'.")

    # Compute matrix of log nearest neighbor distances
    X2 = (X**2).sum(1)

    if n <= mem_threshold:   # speed-memory trade-off
        distance = X2.reshape(-1, 1) + X2 - 2 * np.dot(X, X.T)  # 2x br.cast
        distance.sort(1)
        # Replace invalid values with a small number
        distance[distance < 0] = 1e-7
        knnmatrix = .5 * np.log(distance[:, 1:k2 + 1])
    else:
        knnmatrix = np.zeros((n, k2))
        for i in range(n):
            distance = np.sort(X2[i] + X2 - 2 * np.dot(X, X[i, :]))
            # Replace invalid values with a small number
            distance[distance < 0] = 1e-7
            knnmatrix[i, :] = .5 * np.log(distance[1:k2 + 1])

    # Compute the ML estimate
    S = np.cumsum(knnmatrix, 1)
    indexk = np.arange(k1, k2 + 1)
    dhat = -(indexk - 2) / (S[:, k1 - 1:k2] - knnmatrix[:, k1 - 1:k2] * indexk)

    if estimator == 'levina':
        # Average over estimates and over values of k
        no_dims = dhat.mean()
    if estimator == 'mackay':
        # Average over inverses
        dhat **= -1
        dhat_k = dhat.mean(0)
        no_dims = (dhat_k ** -1).mean()

    return int(no_dims.round())

if __name__ == '__main__':
    m_dim = 100
    n_dim = 2000
    data = np.random.rand(n_dim, m_dim)
    id_ = intrinsic_dimension(data)
    print("Random {}x{} matrix: ID_MLE = {}".format(n_dim, m_dim, id_))
