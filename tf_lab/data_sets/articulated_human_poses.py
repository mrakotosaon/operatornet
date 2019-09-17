'''
Created on Sep 12, 2017

@author: optas
'''

import numpy as np
import os.path as osp

from geo_tool import Point_Cloud, Mesh
from geo_tool.solids import mesh_cleaning as cleaning
from geo_tool.in_out.soup import load_ply


def pc_sampler(mesh_file, n_samples, save_file=None, dtype=np.float32):
    in_mesh = Mesh(file_name=mesh_file)
    in_mesh = cleaning.clean_mesh(in_mesh)
    in_mesh.swap_axes_of_vertices([0, 2, 1])
    ss_points, _ = in_mesh.sample_faces(n_samples)
    pc = Point_Cloud(points=ss_points.astype(dtype))
    pc.center_in_unit_sphere()
    pc, _ = pc.lex_sort()
    if save_file is not None:
        pc.save_as_ply(save_file)
    return pc


def pc_loader(ply_file):
    ending = 'ply'
    drop = len(ending) + 1
    category = ''
    model_id = osp.basename(ply_file)
    model_id = model_id[:-drop]
    return load_ply(ply_file), model_id, category
