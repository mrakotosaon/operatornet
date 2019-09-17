'''
Created on December 23, 2016

@author: optas
'''

import numpy as np


def is_integer(x):
    '''
    4 or 4.0 are considered integers, but 4.2 is not. Also the boolean values [True and False] are considered integers.
    '''
    return np.all(np.equal(np.mod(x, 1), 0))


def is_contiguous(array, min_elem=None, max_elem=None):
    ''' Checks if an array contains all the integers values in the range [min_elem, max_elem]. If one of the two bounds
    is not explicitly defined as input, then the minimum/maximum element in the array is used to check the contiguousness.
    '''
    if np.all(is_integer(array)):
        uvalues = np.unique(array)
        min_elem = uvalues[0] if min_elem is None else min_elem
        max_elem = uvalues[-1] if max_elem is None else max_elem
        n_elems = max_elem - min_elem + 1
        if n_elems != len(uvalues):
            return False
        else:
            return np.all(np.equal(uvalues, np.arange(min_elem, max_elem + 1)))
    else:
        return False
