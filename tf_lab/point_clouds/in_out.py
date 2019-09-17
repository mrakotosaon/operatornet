import warnings
import numpy as np
import os.path as osp
from multiprocessing import Pool

from ..in_out.basics import make_train_validate_test_split
# from geo_tool import Mesh, Point_Cloud
# from geo_tool.in_out.soup import load_crude_point_cloud, load_crude_point_cloud_with_normals
# from geo_tool.in_out.soup import load_mesh_from_file

from general_tools.rla.three_d_transforms import rand_rotation_matrix
from general_tools.in_out.basics import files_in_subdirs

blensor_search_pattern = '0_noisy00000.txt'


def _load_crude_pcloud_and_model_id(f_name):
    tokens = f_name.split('/')
    model_id = tokens[-1].split('_')[0]
    class_id = tokens[-2]
    return load_crude_point_cloud(f_name), model_id, class_id


def load_point_clouds_from_filenames(file_names, n_threads=1, loader=_load_crude_pcloud_and_model_id, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids


def _load_blensor_incomplete_pcloud(f_name):
    points = load_crude_point_cloud(f_name, permute=[0, 2, 1])
    pc = Point_Cloud(points=points)
    pc.lex_sort()
    pc.center_in_unit_sphere()
    tokens = f_name.split('/')
    return pc.points, tokens[-2], tokens[-3]


def _load_crude_pcloud_with_normal_and_model_info(f_name):
    tokens = f_name.split('/')
    model_id = tokens[-1].split('_')[0]
    class_id = tokens[-2]
    return load_crude_point_cloud_with_normals(f_name), model_id, class_id


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def write_model_ids_of_datasets(out_dir, model_ids, r_indices):
    for ind, name in zip(r_indices, ['train', 'val', 'test']):
        with open(osp.join(out_dir, name + '_data.txt'), 'w') as fout:
            for t in model_ids[ind]:
                fout.write(' '.join(t[:]) + '\n')


def apply_augmentations(batch, conf):
        if conf.gauss_augment is not None or conf.z_rotate:
                batch = batch.copy()

        if conf.gauss_augment is not None:
            mu = conf.gauss_augment['mu']
            sigma = conf.gauss_augment['sigma']
            batch += np.random.normal(mu, sigma, batch.shape)

        if conf.z_rotate:  # TODO -> add independent rotations to each object, not one per batch.
            r_rotation = rand_rotation_matrix()
            r_rotation[0, 2] = 0
            r_rotation[2, 0] = 0
            r_rotation[1, 2] = 0
            r_rotation[2, 1] = 0
            r_rotation[2, 2] = 1
            batch = batch.dot(r_rotation)

        return batch


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            assert (type(noise) is np.ndarray)
            if copy:
                self.noisy_point_clouds = noise.copy()
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds[perm]
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            return self.point_clouds[start:end], self.labels[start:end], None
        else:
            return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
        return pc, lb, ns

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = np.vstack((self.point_clouds, other_data_set.point_clouds))

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)

        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = np.vstack((self.noisy_point_clouds, other_data_set.noisy_point_clouds))

        self.num_examples = self.point_clouds.shape[0]

        return self


def shuffle_two_pcloud_datasets(a, b, seed=None):
    n_a = a.num_examples
    n_b = b.num_examples
    frac_a = n_a / (n_a + n_b + 0.0)
    frac_b = n_b / (n_a + n_b + 0.0)

    a = a.point_clouds
    b = b.point_clouds
    joint = np.vstack((a, b))
    _, new_a, new_b = make_train_validate_test_split([joint], train_perc=0, validate_perc=frac_a, test_perc=frac_b, seed=seed)

    new_a = PointCloudDataSet(new_a)
    new_b = PointCloudDataSet(new_b)
    if (new_a.num_examples != n_a) or (new_b.num_examples != n_b):
        warnings.warn('The size of the resulting datasets have changed (+-1) due to rounding.')

    return new_a, new_b
