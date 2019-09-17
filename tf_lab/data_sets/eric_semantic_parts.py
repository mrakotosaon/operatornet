'''
Created on Apr 21, 2017

@author: optas
'''

import os.path as osp
import numpy as np
import glob

from general_tools.in_out.basics import create_dir
from geo_tool import Point_Cloud
from geo_tool.in_out.soup import load_crude_point_cloud

# Extensions Eric used.
erics_seg_extension = '.seg'
erics_points_extension = '.pts'

eric_part_pattern = '__?__.ply'    # Function 'equisample_parts_via_bootstrap' saved the parts with this pattern as ending.


all_syn_ids = ['02691156', '02773838', '02954340', '02958343', '03001627', '03261776', '03467517',
               '03624134', '03636649', '03790512', '03797390', '03948459', '04099429', '04225987',
               '04379243']


def _prepare_io(data_top_dir, out_top_dir, synth_id, boot_n):
    points_top_dir = osp.join(data_top_dir, synth_id, 'points')
    segs_top_dir = osp.join(data_top_dir, synth_id, 'expert_verified', 'points_label')
    original_density_dir = osp.join(out_top_dir, synth_id, 'original_density')
    bstrapped_out_dir = osp.join(out_top_dir, str(boot_n) + '_bootstrapped', synth_id)
    create_dir(original_density_dir)
    create_dir(bstrapped_out_dir)
    return segs_top_dir, points_top_dir, original_density_dir, bstrapped_out_dir


# def eric_annotated(data_top_dir, out_top_dir, synth_id, boot_n=2048, dtype=np.float32):
#     ''' Writes out point clouds with a segmentation mask according to Eric's annotation.
#     The point clouds are 1) the original point clouds that Eric sampled  2) a bootstrapped version of them.
#     '''
# 
#     segs_top_dir, points_top_dir, original_density_dir, bstrapped_out_dir = _prepare_io(data_top_dir, out_top_dir, synth_id, boot_n)
# 
#     erics_seg_extension = '.seg'      # Extensions Eric used.
#     erics_points_extension = '.pts'
# 
#     for file_name in glob.glob(osp.join(segs_top_dir, '*' + erics_seg_extension)):
#         model_name = osp.basename(file_name)[:-len(erics_seg_extension)]
#         pt_file = osp.join(points_top_dir, model_name + erics_points_extension)
#         points = load_crude_point_cloud(pt_file, permute=[0, 2, 1])
#         n_points = points.shape[0]
# 
#         pc = Point_Cloud(points=points.astype(dtype))
# #         pc = pc.center_in_unit_sphere()
#         pc, lex_index = pc.lex_sort()
# 
#         gt_seg = np.loadtxt(file_name, dtype=np.float32)
#         gt_seg = gt_seg[lex_index]
#         gt_seg = gt_seg.reshape((n_points, 1))
#         seg_ids = np.unique(gt_seg)
#         if seg_ids[0] == 0:
#             seg_ids = seg_ids[1:]   # Zero is not a real segment.
# 
#         header_str = 'erics-annotated_segs\nseg_ids=%s' % (str(seg_ids.astype(np.int)).strip('[]'))
#         out_data = np.hstack((pc.points, gt_seg))
#         out_file = osp.join(original_density_dir, model_name + segs_ext)
#         np.savetxt(out_file, out_data, header=header_str)
#         boot_strap_lines_of_file(out_file, boot_n, osp.join(bstrapped_out_dir, model_name + segs_ext), skip_rows=2)


def equisample_parts_via_bootstrap(data_top_dir, out_top_dir, synth_id, n_samples=2048, dtype=np.float32):
    ''' For each object in the synth_id extract the parts and bootstrap them into point-clouds with n_samples each.
    '''
    segs_top_dir, points_top_dir, _, bstrapped_out_dir = _prepare_io(data_top_dir, out_top_dir, synth_id, n_samples)

    for file_name in glob.glob(osp.join(segs_top_dir, '*' + erics_seg_extension)):
        model_name = osp.basename(file_name)[:-len(erics_seg_extension)]
        pt_file = osp.join(points_top_dir, model_name + erics_points_extension)
        points = load_crude_point_cloud(pt_file, permute=[0, 2, 1], dtype=dtype)
        gt_seg = np.loadtxt(file_name, dtype=np.float32)
        seg_ids = np.unique(gt_seg)
        if seg_ids[0] == 0:
            seg_ids = seg_ids[1:]   # Zero is not a real segment.

        for seg in seg_ids:
            pc = Point_Cloud(points=points[gt_seg == seg])
            pc, _ = pc.sample(n_samples)
            pc, _ = pc.lex_sort()
            seg_token = '__' + str(seg) + '__'
            out_file = osp.join(bstrapped_out_dir, model_name + seg_token)
            pc.save_as_ply(out_file)


def part_pc_loader(ply_file):
    pc = Point_Cloud(ply_file=ply_file)
    tokens = ply_file.split('/')
    model_id = tokens[-1][:-len(eric_part_pattern)]
    part_id = tokens[-1][-len(eric_part_pattern):-(len('.ply'))]
    syn_id = tokens[-2]
    return pc.points, (model_id, part_id), syn_id
