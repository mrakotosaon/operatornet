'''
Created on Aug 11, 2017

@author: optas
'''


import glob
import os.path as osp
import numpy as np
from collections import Counter

from general_tools.in_out.basics import files_in_subdirs
from general_tools.simpletons import are_disjoint_sets


def read_saved_epochs(saved_dir):
    epochs_saved = []
    files = glob.glob(osp.join(saved_dir, 'models.ckpt-*.index'))
    for f in files:
        epochs_saved.append(int(osp.basename(f)[len('models.ckpt-'):-len('.index')]))
        epochs_saved.sort()
    return epochs_saved


def make_train_validate_test_split(arrays, train_perc=0, validate_perc=0, test_perc=0, shuffle=True, seed=None):
    ''' This is a memory expensive operation since by using slicing it copies the input arrays.
    '''

    if not np.allclose((train_perc + test_perc + validate_perc), 1.0):
        raise ValueError()

    if type(arrays) is not list:
        arrays = [arrays]

    n = arrays[0].shape[0]   # n examples.
    if len(arrays) > 1:
        for a in arrays:
            if a.shape[0] != n:
                raise ValueError('All arrays must have the same number of rows/elements.')

    index = np.arange(n)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(index)
    else:
        perm = np.arange(n)

    train_end = int(train_perc * n)
    validate_end = int(validate_perc * n) + train_end

    train_data = []
    validate_data = []
    test_data = []
    r_ind = (perm[:train_end], perm[train_end:validate_end], perm[validate_end:])

    for a in arrays:
        train_data.append(a[r_ind[0]])
        validate_data.append(a[r_ind[1]])
        test_data.append(a[r_ind[2]])

    if len(train_data) == 1:    # Single array split
        return train_data[0], validate_data[0], test_data[0], r_ind
    else:
        return train_data, validate_data, test_data, r_ind


class Data_Splitter():
    '''currently works with shape-net data in mind: xx/syn_id/datum
    '''

    def __init__(self, top_data_path, data_file_ending, random_seed=42):
        self.top_data_path = top_data_path
        self.random_seed = random_seed
        self.files_ending = data_file_ending

    def make_splits(self, syn_id, tr_val_te_loads, keep_only=None):
        ''' works for single syn_id
        '''
        syn_data_dir = osp.join(self.top_data_path, syn_id)
        all_file_names = [f for f in files_in_subdirs(syn_data_dir, self.files_ending)]
        all_file_names = [osp.basename(f)[:-len(self.files_ending)] for f in all_file_names]

        tr, val, te, _ = make_train_validate_test_split(np.array(all_file_names, dtype=object),
                                                        train_perc=tr_val_te_loads[0],
                                                        validate_perc=tr_val_te_loads[1],
                                                        test_perc=tr_val_te_loads[2], seed=self.random_seed)

        assert(are_disjoint_sets([set(tr), set(te), set(val)]))
        assert(set(tr).union(set(val)).union(set(te)) == set(all_file_names))

        if keep_only is not None:
            tr = np.random.choice(tr, min(len(tr), keep_only[0]), replace=False)
            val = np.random.choice(val, min(len(val), keep_only[1]), replace=False)
            te = np.random.choice(te, min(len(te), keep_only[2]), replace=False)

        return tr, val, te

    def write_splits(self, out_file, model_names, syn_id):
        with open(out_file, 'a') as fout:
            for model_name in model_names:
                fout.write(model_name + ' ' + syn_id + '\n')

    def load_splits(self, in_file, full_path=True):
        file_names = []
        with open(in_file, 'r') as fin:
            for example in fin:
                model_name, syn_id = example.split()
                if full_path:
                    full_file = osp.join(self.top_data_path, syn_id, model_name + self.files_ending)
                else:
                    full_file = syn_id + '_' + model_name
                file_names.append(full_file)
        return file_names

    def generate_splits(self, syn_ids, top_out_dir, tr_val_te_loads, keep_only=None, run_debug=False):
        for syn_id in syn_ids:
            tr, val, te = self.make_splits(syn_id, tr_val_te_loads, keep_only)
            self.write_splits(osp.join(top_out_dir, 'train.txt'), tr, syn_id)
            self.write_splits(osp.join(top_out_dir, 'val.txt'), val, syn_id)
            self.write_splits(osp.join(top_out_dir, 'test.txt'), te, syn_id)

        if run_debug:
            tr = self.load_splits(osp.join(top_out_dir, 'train.txt'), False)
            te = self.load_splits(osp.join(top_out_dir, 'test.txt'), False)
            val = self.load_splits(osp.join(top_out_dir, 'val.txt'), False)

            print 'are disjoint sets?', are_disjoint_sets([set(tr), set(te), set(val)])

            print 'distributions'
            for d, l in zip([tr, te, val], ['train', 'test', 'val']):
                cnt = Counter()
                temp = [i.split('_') for i in d]
                for w in temp:
                    cnt[w[0]] += 1
                print l, cnt

            print 'are unique?', len(set(tr)) == len(tr) and len(set(val)) == len(val) and len(set(te)) == len(te)
