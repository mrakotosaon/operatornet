'''
Created on August 26, 2017

@author: optas
'''

import numpy as np
from numpy import dtype


def index_in_condensed_dist_matrix(i, j, n_points):
    '''As is customary, functions computing pairwise distances (like scipy.spatial.pdist) return an upper triangular
    distance matrix in a condensed form to save space. This function given the (i,j) indices of the elements
    for which the distance is requested, computes the corresponding index in the condensed distance matrix.
    Input:
        i - (int) row index (i-th element)
        j - (int) column index (j-th element)
        n_points - (int) the total number of points over which the pairwise distances were computed.
    '''
    if i == j:
        raise ValueError('Condensed Matrix does not carry self-distances, which are assumed to be zero.')
    if i > j:
        index = n_points * j - j * (j + 1) / 2 + i - 1 - j
    else:
        index = n_points * i - i * (i + 1) / 2 + j - 1 - i

    return index


def condensed_form_to_square_form(condensed_data, square_dim, dtype=np.float32):
    ''' does it much faster than numpy.squareform
    '''
    square = np.zeros(shape=(square_dim, square_dim), dtype=dtype)
    square[np.tril_indices(square_dim, -1)] = condensed_data
    square = square + square.T
    return square


def incremental_farthest_sampling(all_pdists, k, exluded_points=None, seed=None):
    '''Returns k points (unique indices) that are picked in a greedy way
    which attempts to maximize their minimum pairwise distances. The seed controls the
    resulting indices.
    '''
    remaining_points = np.arange(all_pdists.shape[0])

    if exluded_points is not None:
        remaining_points = np.setdiff1d(remaining_points, exluded_points)

    if seed is not None:
        np.random.seed(seed)

    solution_set = [np.random.choice(remaining_points, 1)[0]]
    remaining_points = np.setdiff1d(remaining_points, solution_set, assume_unique=True)

    for _ in range(k - 1):
        max_dist = -1
        next_point = -1

        for p in remaining_points:
            min_d = np.min(all_pdists[p, solution_set])
            if min_d > max_dist:
                max_dist = min_d
                next_point = p

        solution_set.append(next_point)
        remaining_points = np.setdiff1d(remaining_points, next_point, assume_unique=True)

    assert(len(np.unique(solution_set)) == k)
    assert(len(solution_set) == k)
    assert(len(np.intersect1d(remaining_points, np.array(solution_set))) == 0)

    return solution_set


def evaluate_solution(solution_set):

    distance = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

    return sum([distance(a, b) for a, b in zip(solution_set[:-1], solution_set[1:])])

