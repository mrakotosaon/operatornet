'''
Created on Mar 32, 2017

@author: optas

Associate code and data for manipulating the shapes of Shape-Net.

'''
import six
import numpy as np
import os.path as osp

from geo_tool.in_out.soup import load_ply
from geo_tool import Mesh, Point_Cloud
import geo_tool.solids.mesh_cleaning as cleaning
from geo_tool.in_out.soup import load_crude_point_cloud
from general_tools.in_out.basics import files_in_subdirs, create_dir

from .. point_clouds.in_out import load_point_clouds_from_filenames, PointCloudDataSet

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

# '02858304' (boat) and '02834778': 'bicycle', don't exist in SN-Core V2.


def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def pc_uniform_sampler(mesh_file, n_samples, swap_y_z=True, save_file=None, dtype=np.float32, out_folder=None):
    ''' Given a mesh, it computes a point-cloud that is uniformly sampled
    from its area elements.
    '''
    in_mesh = Mesh(file_name=mesh_file)
    if swap_y_z:
        in_mesh.swap_axes_of_vertices([0, 2, 1])
    in_mesh = cleaning.clean_mesh(in_mesh)
    ss_points, _ = in_mesh.sample_faces(n_samples)
    pc = Point_Cloud(points=ss_points.astype(dtype))
    pc.center_in_unit_sphere()
    pc, _ = pc.lex_sort()
    if save_file is not None:
        pc.save_as_ply(save_file)
    return pc


def pc_loader(f_name):
    '''Assumes that the point-clouds were created with:
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id


def fps_sampled_loader(in_f, save_dir=None):
    ''' Loads pc's created with Matlab\'s code and FPS sampling.
    '''
    pc = load_crude_point_cloud(in_f)
    pc = Point_Cloud(pc).permute_points([0, 2, 1]).points
    syn_id = in_f.split('/')[-3]
    model_name = in_f.split('/')[-2]

    pc = Point_Cloud(pc)
    pc.center_axis()
    pc.center_in_unit_sphere()
    pc, _ = pc.lex_sort()
    if save_dir is not None:
        out_dir = osp.join(save_dir, syn_id)
        create_dir(out_dir)
        out_file = osp.join(out_dir, model_name)
        pc.save_as_ply(out_file)

    return pc, model_name, syn_id


def load_all_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply', verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, verbose=verbose)
    return PointCloudDataSet(pclouds, labels=syn_ids + '_' + model_ids, init_shuffle=False)
