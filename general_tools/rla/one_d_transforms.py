'''
Created on January 17, 2017

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
from scipy.stats import mode


def smooth_normal_outliers(array, dev):
    '''In each row of the input array finds outlier elements and transforms their values.
    An outlier in row[i], is any element of that row that is in magnitude bigger than
    \mu[i] + `dev` times \sigma[i], where \mu[i] is the mean value
    and \sigma the standard deviation of the values of row i.

    Note: It changes the array inline.
    '''

    stds = np.std(array, axis=1)
    means = np.mean(array, axis=1)
    line_elems = np.arange(array.shape[1])
    for i in xrange(array.shape[0]):
        outliers = abs(array[i]) > dev * stds[i] + means[i]
        inliers = np.setdiff1d(line_elems, outliers, assume_unique=False)
        mu_i = np.mean(array[i, inliers])
#         array[i, outliers] = means[i]
        array[i, outliers] = mu_i
    return array


def find_non_homogeneous_vectors(array, thres):
    '''
    '''
    index = []
    n = float(array.shape[1])
    for i, vec in enumerate(array):
        frac = mode(vec)[1][0] / n
        if frac < thres:
            index.append(i)
    return index
