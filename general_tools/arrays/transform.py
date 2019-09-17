'''Created on December 23, 2016

@author: optas
'''

import numpy as np
from . import is_true


def make_contiguous(array, start=0):
    ''' The array will be transformed inline, to contain integers from [start, start+max_i) where max_i is the number of unique integers it contains.
    The relative order of the integers will remain.
    '''

    if not is_true.is_integer(array):
        raise ValueError('Cannot transform an non integer array to be contiguous.')

    uvalues = np.unique(array)
    d = {key: value + start for (value, key) in enumerate(uvalues)}
    for i, _ in enumerate(array):     # TODO -> use flatten to work on all dimensions
        array[i] = d[array[i]]

    return array
