'''
Created on November 8, 2016

@author: optas
'''

import numpy as np
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
import itertools
from numpy import dtype

from .. arrays.is_true import is_contiguous, is_integer 

def bench_clustering(estimator, name, gt_labels, verbose=True):
    '''Compares a clustering produced by an algorithm like kmeans or spectral clustering
    with a ground-truth clustering under some popular fitness metrics.
    '''    
    if verbose:
        print ('%30s   %4s   %4s   %4s   %4s   %4s' % (' ', 'Homog', 'Compl', 'V-Mea', 'A-Ran', 'A-Mut') )

    print('%30s   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name,
             metrics.homogeneity_score(gt_labels, estimator.labels_),
             metrics.completeness_score(gt_labels, estimator.labels_),
             metrics.v_measure_score(gt_labels, estimator.labels_),
             metrics.adjusted_rand_score(gt_labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(gt_labels,  estimator.labels_),)
         )

def kmeans_cluster_pseudo_similarity(estimator):
    # similarity is not symmetric. also it is bound in 0 to 1.
    cluster_dists = squareform(pdist(estimator.cluster_centers_))
    cluster_dists = (cluster_dists.T / np.max(cluster_dists, 0)).T   # For each cluster max distance is 1    
    cluster_sims = 1.0 - cluster_dists
#     cluster_probs = (cluster_sims.T / np.sum(cluster_sims, 0)).T
    return cluster_sims

def spectral_cluster_pseudo_similarity(estimator):
    labels = estimator.labels_
    class_ids = np.unique(labels)
    n_classes = len(class_ids)
    similarity = np.zeros((n_classes, n_classes))
    affinity = estimator.affinity_matrix_ 
    for c1, c2 in itertools.combinations(class_ids, 2):
        # Sum edge weights across clusters (double counts an edge)
        similarity[c1, c2] = np.sum((labels == c1) * affinity * (labels == c2).T) / 2.0
        
    for c in range(n_classes):
        # Sum edge weights within each cluster (double counts an edge)
        similarity[c, c] = np.sum((labels == c) * affinity * (labels == c).T) / 2.0
        
    similarity += similarity.T
    similarity = (similarity.T / np.max(similarity, 0)).T   # For each cluster max similarity is 1    
    
    return similarity

def clustering_indicator_vectors(labels):
    n_points = len(labels)    
    n_clusters = len(np.unique(labels)) # Assert contiguous
    
    if not is_contiguous(labels, min_elem=0, max_elem=n_clusters-1):
        raise NotImplementedError()
    
    indicators = np.zeros((n_clusters, n_points), dtype = np.float32)    
    for i, label in enumerate(labels):
        indicators[label,i] = 1
      
    for i in range(n_points):
        assert(np.all(np.where(indicators[:,i] == 1) == labels[i]))
    
    return indicators