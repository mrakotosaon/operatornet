'''
Created on Jan 9, 2018

@author: optas
'''


import numpy as np
from scipy.io import loadmat
from helper import load_gt_latent_params, load_pclouds_of_shapes, sub_collection_pose_labels, prepare_train_test_val, make_data

from sklearn.cluster import KMeans


def load_synced_bases(synced_bases_file, n_cons, n_shapes, debug=False):
    synced_bases = loadmat(synced_bases_file)
    synced_bases = synced_bases['sync_bases']
    synced_bases = synced_bases[:, :n_cons]
    if debug:
        temp = []
        n_evecs = int(synced_bases.shape[0] / n_shapes)
        for i in range(0, synced_bases.shape[0], n_evecs):
            temp.append(synced_bases[i:i + n_evecs])
        temp = np.array(temp)
        assert(np.all(synced_bases.reshape((n_shapes, -1, n_cons)) == temp))
    return synced_bases.reshape((n_shapes, -1, n_cons))

def load_latent_diff(latent_diff_file, n_dim, n_shapes):
    latent_diff = loadmat(latent_diff_file)
    latent_diff = latent_diff['latent_diff']
    latent_diff = latent_diff[:, :n_dim]

    return latent_diff.reshape((n_shapes, -1, n_dim))

def load_ruqi_sanity_maps(sanity_file, map_type):
    res = loadmat(sanity_file)
    res = res[map_type]
    n_shapes = len(res)
    diff_dims = res[0][0].shape
    temp = np.zeros(shape=(n_shapes, ) + diff_dims)
    for i in xrange(n_shapes):
        temp[i] = res[i][0]
    res = temp
    return res


def produce_diff_maps(synced_bases_file, n_cons, n_shapes):
    diff_shape = (n_cons, n_cons)
    s_bases = load_synced_bases(synced_bases_file, n_cons, n_shapes)
    in_diffs = np.zeros(shape=((n_shapes,) + diff_shape))
    for i in xrange(n_shapes):
        in_diffs[i] = s_bases[i].T.dot(s_bases[i])
    return in_diffs


def raw_data(top_mesh_dir, gt_param_f, sub_member_per_class, n_pc_points, norm_pc=False):
    gt_latent_params = load_gt_latent_params(gt_param_f, sub_member_per_class)
    in_pcs = load_pclouds_of_shapes(top_mesh_dir, sub_member_per_class, n_pc_points, normalize=norm_pc)
    pose_labels = sub_collection_pose_labels(sub_member_per_class)
    return gt_latent_params, in_pcs, pose_labels


def classes_of_tasks(task):
    if task == 'pose_clf':
        n_class = 8
    elif task == 'unsup_clf' or task == 'regression':
        n_class = 12
    return n_class


def prep_splits_labels_for_task(task, gt_latent_params, pose_labels, train_per, test_per, seed=None):
    if task == 'pose_clf':
        labels = pose_labels
    elif task == 'unsup_clf':
        n_classes = gt_latent_params.shape[1]
        assert(classes_of_tasks(task) == n_classes)
        unsup_clf = KMeans(n_clusters=n_classes, random_state=seed)
        labels = unsup_clf.fit_predict(gt_latent_params)
    elif task == 'regression':
        labels = gt_latent_params

    n_shapes = len(gt_latent_params)

    if task.endswith('clf'):
        splits = prepare_train_test_val(n_shapes, labels, train_per, test_per, seed=seed, stratify=True)
    else:
        splits = prepare_train_test_val(n_shapes, labels, train_per, test_per, seed=seed)

    return splits, labels


def produce_net_data(in_pcs, splits, labels, diff_maps, use_pc, norm_diffs=True):
    if use_pc:
        feeds = in_pcs
    else:
        feeds = diff_maps

    data = make_data(splits, feeds, labels)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    if not use_pc and norm_diffs:
        diff_mu = np.mean(train_data.feed, axis=0)
        train_data.feed -= diff_mu
        test_data.feed -= diff_mu
        val_data.feed -= diff_mu
    return data
