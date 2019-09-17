from general_tools.notebook.gpu_utils import setup_one_gpu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('use_pc', type=int)
parser.add_argument('n_cons', type=int)
parser.add_argument('arch', type=str)
parser.add_argument('knn', type=int)
parser.add_argument('seed', type=int)
parser.add_argument('--GPU', type=int, default=0, required=False)

opt = parser.parse_args()
setup_one_gpu(opt.GPU)

import numpy as np
import tensorflow as tf
import os.path as osp
import matplotlib.pylab as plt
from general_tools.notebook.tf import reset_tf_graph
from geo_tool import Point_Cloud, Mesh
from geo_tool.solids.plotting import plot_mesh_via_matplotlib as plot_mesh
import helper
from tf_lab.diff_maps.in_out import raw_data, produce_net_data,prep_splits_labels_for_task, produce_diff_maps, classes_of_tasks
from tf_lab.diff_maps.basic_nets import pc_net, diff_mlp_net, pc_versions
from tf_lab.diff_maps.basic_nets import Basic_Net

if opt.use_pc == 1:
    use_pc = True
elif opt.use_pc == 0:
    use_pc = False
else:
    assert(False)

arch =  opt.arch
n_cons = opt.n_cons
knn = opt.knn
seed = opt.seed

top_mesh_dir = '/orions4-zfs/projects/optas/DATA/Meshes/SCAPE_8_poses_2/'
gt_param_f = osp.join(top_mesh_dir, 'gt_shape_params.mat')
total_shapes = helper.total_shapes
n_pose_classes = helper.n_pose_classes
top_data_dir = '/orions4-zfs/projects/optas/DATA/OUT/latent_diff_maps/experiments/SCAPE_8_poses_2'

synced_bases_file = osp.join(top_data_dir, '50_extract_%d_knn_50_fmapd.mat' % (knn,) )
sub_member_per_class = 50
n_shapes = sub_member_per_class * n_pose_classes
seed = None
val_per = 0.10
test_per = 0.15
train_per = 1.0 - (val_per + test_per)
n_pc_points = 1024
task = 'regression'
mean_norm_diffs = True
n_reps = 3

if use_pc:
#   [learning_rate = 0.001, 0.002, 0.005, 0.01, 0.05]
    learning_rate = 0.005
    batch_size = 50
    n_epochs = 500
else:
    learning_rate = 0.007
    batch_size = 50
    n_epochs = 500

reset_tf_graph()
tf.set_random_seed(seed)

if use_pc:
    n_filters, n_neurons = pc_versions('v2')
    net_out, feed_pl, label_pl = pc_net(n_pc_points, task, n_filters, n_neurons)
else:
    net_out, feed_pl, label_pl = diff_mlp_net(n_cons, task)    
    
net = Basic_Net(net_out, feed_pl, label_pl)
net.define_loss(task)
net.define_opt(learning_rate)
net.start_session()

gt_latent_params, in_pcs, pose_labels = raw_data(top_mesh_dir, gt_param_f, sub_member_per_class, n_pc_points)
diff_maps = produce_diff_maps(synced_bases_file, n_cons, n_shapes)

splits, labels = prep_splits_labels_for_task(task, gt_latent_params, pose_labels, train_per, test_per, seed)

net_data = produce_net_data(in_pcs, splits, labels, diff_maps, use_pc, mean_norm_diffs)

n_classes = classes_of_tasks(task)

#s_over_reps = []
n_epochs = 500
verbose = False

for _ in range(n_reps):
    net.sess.run(net.init)    
    stats = net.train(n_epochs, batch_size, net_data, task, verbose=verbose)    
    n_opt, val_maximizer, gen_error, test_best = score_net(task, stats)    
    print n_opt, val_maximizer, gen_error, test_best    
    #s_over_reps.append((n_opt, val_maximizer, gen_error, test_best))