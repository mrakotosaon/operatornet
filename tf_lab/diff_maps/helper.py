import numpy as np
import hdf5storage
import os.path as osp
import matplotlib.pylab as plt
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

#from geo_tool import Point_Cloud, Mesh
from tf_lab.data_sets.numpy_dataset import NumpyDataset

'''
Funcs to deal with SCAPE_8_pose collection.
'''

total_shapes = 2000
members_per_pose_class = 250
n_pose_classes = 8
dfaust = False

def sub_collection_indices(sub_size_per_class):
    sub_size = sub_size_per_class * n_pose_classes
    original_idx = np.zeros(sub_size, dtype=int)
    c = 0
    for i in range(0, total_shapes, members_per_pose_class):
        for j in range(sub_size_per_class):
            original_idx[c] = i + j
            c += 1
    return original_idx


def sub_collection_pose_labels(sub_size_per_class):
    n_sub = sub_size_per_class * n_pose_classes
    pose_labels = np.zeros(n_sub)
    c = -1
    for i in range(n_sub):
        if i % sub_size_per_class == 0:
            c += 1
        pose_labels[i] = c
    return pose_labels

def sub_collection_indices_new():
    import random
    import numpy
    ids = [i for i in range(128*20 -1)]
    selected_ids = []
    n_train = 2000
    n_test = 2500
    random.seed(42)
    for i in range(n_train//128):
        for j in range(128):
            el = random.choice(ids[(j*20):((j+1)*20)])
            while el in selected_ids:
                el = random.choice(ids[(j*20):((j+1)*20)])
            selected_ids.append(el)
    while len(selected_ids)<n_train:
        el = random.choice(ids)
        if el not in selected_ids:
            selected_ids.append(el)
    random.shuffle(selected_ids)
    while len(selected_ids)<n_test:
        el = random.choice(ids)
        if el not in selected_ids:
            selected_ids.append(el)
    return selected_ids

def sub_collection_indices_new2():
    import random
    import numpy
    ids = [i for i in range(128*20)]
    selected_ids = []
    n_train = 2000
    n_test = 2500
    random.seed(42)
    for i in range(n_train//128):
        for j in range(128):
            el = random.choice(ids[(j*20):((j+1)*20)])
            while el in selected_ids:
                el = random.choice(ids[(j*20):((j+1)*20)])
            selected_ids.append(el)
    while len(selected_ids)<n_train:
        el = random.choice(ids)
        if el not in selected_ids:
            selected_ids.append(el)
    random.shuffle(selected_ids)

    unused_ids = [x for x in ids if x not in selected_ids]
    random.shuffle(unused_ids)
    selected_ids += unused_ids

    return selected_ids

# def sub_collection_pos_labels_new():


def load_pclouds_of_shapes(top_data_dir, sub_size_per_class, n_pc_points, normalize=False):
    if not dfaust:
        in_pcs = osp.join(top_data_dir, 'uniform_point_clouds_%d_pts.npz' % (n_pc_points, ))
        in_pcs = np.load(in_pcs)
    else:
        in_pcs = "/home/marie-julie/Dropbox/Data_transition/D_FAUST/sample_points_{}.mat".format(n_pc_points)
        in_pcs = loadmat(in_pcs)
    in_pcs = in_pcs[in_pcs.keys()[0]]
    print(in_pcs.shape)
    if dfaust:
        in_pcs = np.reshape(in_pcs, (-1, n_pc_points, 3))
        print('pc shape', in_pcs.shape)

    if not dfaust:
        idx = sub_collection_indices(sub_size_per_class)
    else:
        idx = sub_collection_indices_new()
    in_pcs = in_pcs[idx]
    if normalize:
        res = np.zeros_like(in_pcs)
        for i, pts in enumerate(in_pcs):
            pc = Point_Cloud(pts).center_in_unit_sphere()
            res[i] = pc.points
    else:
        res = in_pcs
    return res



def load_gt_latent_params(gt_file, sub_size_per_class):
    gt_latent_params = osp.join(gt_file)
    gt_latent_params = loadmat(gt_latent_params)
    gt_latent_params = gt_latent_params['parammat'].T
    original_idx = sub_collection_indices(sub_size_per_class)
    return gt_latent_params[original_idx]


def load_meshes(mesh_dir, mesh_ids):
    meshes = []
    for i in mesh_ids:
        in_f = osp.join(mesh_dir, 'Shape%s.off' % (i,))
        in_m = Mesh(file_name=in_f)
        meshes.append(in_m)
    return meshes


def prepare_train_test_val(n_shapes, class_labels, train_per, test_per, seed=None, stratify=None):
    all_ids = np.arange(n_shapes)

    if stratify is not None:
        stratify = class_labels

    train_ids, rest_ids = train_test_split(all_ids, stratify=stratify, train_size=train_per, test_size=1.0 - train_per, random_state=seed)

    if stratify is not None:
        stratify = class_labels[rest_ids]

    ts = int(n_shapes * test_per)
    test_s = len(rest_ids) - ts
    test_ids, val_ids = train_test_split(rest_ids, stratify=stratify, train_size=ts, test_size=test_s, random_state=seed)
    in_data = dict()
    in_data['train'] = train_ids
    in_data['test'] = test_ids
    in_data['val'] = val_ids
    return in_data


def make_data(in_data, in_feeds, class_labels):
    res = dict()
    for s in ['train', 'test', 'val']:
        idx = in_data[s].copy()
        res[s] = NumpyDataset([in_feeds[idx], class_labels[idx], idx], ['feed', 'labels', 'ids'], init_shuffle=False)
    return res


### DELETE Below
def load_diff_maps(in_file, zero_thres):
    in_diffs = hdf5storage.loadmat(in_file)
    n_shapes = len(in_diffs['ucb'])
    diff_dims = in_diffs['ucb'][1][0].shape
    temp = np.zeros(shape=(n_shapes, ) + diff_dims )
    for i in xrange(n_shapes):
        temp[i] = in_diffs['ucb'][i][0]
    in_diffs = temp
    return in_diffs
