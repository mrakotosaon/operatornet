'''
Created on February 24, 2017

@author: optas
'''

import operator
import numpy as np


def sort_dict_by_key(in_dict, reverse=False):
    return sorted(in_dict.items(), key=operator.itemgetter(0), reverse=reverse)


def sort_dict_by_val(in_dict, reverse=False):
    return sorted(in_dict.items(), key=operator.itemgetter(1), reverse=reverse)


def invert_dictionary(d):
    inv_map = {v: k for k, v in d.iteritems()}
    return inv_map


def merge_two_dicts(x, y):
    z = x.copy()   # Start with x's keys and values.
    z.update(y)    # Modifies z with y's keys and values & returns None.
    return z


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def select_first_last_and_k(in_list, k):
    '''Select the first and last element of a list among exactly k elements equally spaced
    in the in_list[1:-1]
    '''

    f = in_list[0]
    e = in_list[-1]
    index = np.floor(np.linspace(1, len(in_list) - 2, k)).astype(np.int16)
    res = [in_list[i] for i in index]
    res.insert(0, f)
    res.append(e)
    return res


def indices_in_iterable(target, queries):
    '''Find index of each item of the 'queries' in the 'target'.
    If a query does not exist in the target, the corresponding index is set to -1.
    '''

    if len(np.unique(np.array(target, dtype=object))) != len(target):
        raise ValueError('Target has to be comprised by unique elements.')

    d = {name: i for i, name in enumerate(target)}
    mapping = []
    for name in queries:
        if name not in d:
            mapping.append(-1)
        else:
            mapping.append(d[name])
    mapping = np.array(mapping)
    return np.array(mapping)


def are_disjoint_sets(sets):
    union = set()
    for s in sets:
        for x in s:
            if x in union:
                return False
            union.add(x)
    return True
