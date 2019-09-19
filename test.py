# from general_tools.notebook.gpu_utils import setup_one_gpu
# setup_one_gpu(1)
from train import load_SD, diff_reconstructor, align
import tensorflow as tf
import numpy as np
import os
from general_tools.notebook.tf import reset_tf_graph
from plyfile import PlyData, PlyElement
from scipy.io import loadmat
from scipy.linalg import logm, expm


def totuple(a):
    return [ tuple(i) for i in a]

def save_ply(V,T,filename):
    vertex = np.array(totuple(V),dtype=[('x', 'f4'), ('z', 'f4'),('y', 'f4')])
    face = np.array([ tuple([i]) for i in T],dtype=[('vertex_indices', 'i4', (3,))])
    el1 = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el1,el2]).write(filename)

def reconstruct_shapes(sess, ops, SD, results_path):
    triv = loadmat("data/auxdata/TRIV.mat")['TRIV'] - 1
    feed_dict = {ops['feed_pl']:SD}
    recons_val = sess.run(ops['x_reconstr'], feed_dict)
    for i in range(len(recons_val)):
        save_ply(recons_val[i], triv,os.path.join(results_path, "recons_shape{}.ply".format(i)) )

def interpolate_shapes(sess, ops, SD, results_path, n_interpolation = 10):
    # interpolate last 2 shapes in SD
    triv = loadmat("data/auxdata/TRIV.mat")['TRIV'] - 1
    logA = [logm(SD[3, : ,:,i]) for i in range(3)]
    logB = [logm(SD[4, : ,:,i]) for i in range(3)]

    interpolated_sd = [np.expand_dims(np.array([expm((1-t)*logA[sd_type] + t*logB[sd_type]) for t in np.linspace(0,1,n_interpolation)]), -1) for sd_type in range(3)]
    interpolated_sd = np.concatenate(interpolated_sd, -1)
    feed_dict = {ops['feed_pl']:interpolated_sd}
    recons_val = sess.run(ops['x_reconstr'], feed_dict)
    for i in range(n_interpolation):
         save_ply(recons_val[i], triv,os.path.join(results_path, "interp_shape{}.ply".format(i)) )

def analogy(sess, ops, SD, results_path):
    # analogy between first 3 shapes
    triv = loadmat("data/auxdata/TRIV.mat")['TRIV'] - 1
    SD_X = [np.expand_dims(np.dot(np.dot(SD[1, :, :, i], np.linalg.pinv(SD[0, :, :, i])), SD[2, :, :, i]), -1) for i in range(3)]
    SD_X = np.concatenate(SD_X, 2)
    eval_sd = np.concatenate([SD[:3], np.expand_dims(SD_X, 0)], 0)
    feed_dict = {ops['feed_pl']:eval_sd}
    recons_val = sess.run(ops['x_reconstr'], feed_dict)
    for i, e in enumerate(["A", "B", "C", "X"]):
        save_ply(recons_val[i], triv,os.path.join(results_path, "analogy_shape{}.ply".format(e)) )

def init_graph(n_cons, n_pc_points, n_channels, model_path):
    # init graph
    reset_tf_graph()
    x_reconstr, feed_pl, labels_pl = diff_reconstructor(n_cons, n_pc_points, n_channels)
    loss = tf.losses.mean_squared_error(align(labels_pl, x_reconstr), x_reconstr)
    ops = {"loss": loss, "x_reconstr": x_reconstr, "feed_pl": feed_pl, "labels_pl": labels_pl}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    return sess, ops

def load_data(data_path, n_shapes, n_cons):
    area = load_SD(os.path.join(data_path, "AreaSD.mat"), n_shapes, n_cons)
    conf = load_SD(os.path.join(data_path, "ConfSD.mat"), n_shapes, n_cons)
    ext = load_SD(os.path.join(data_path, "ExtSD.mat"), n_shapes, n_cons)
    data = np.concatenate([ext, area, conf], 3)
    return data


if __name__ == '__main__':
    n_pc_points = 1000
    n_cons = 60 # size of reduced basis
    n_channels = 3
    model_path = "models/best_surreal+dfaust_red_conv_A+C+E.ckpt" # demo model path
    #model_path =  "models/best_model_{channels}.ckpt".format(channels = channels)
    results_path = "results"
    data_path = "data/shapeoperators"
    n_shapes = 5
    sess, ops = init_graph(n_cons, n_pc_points, n_channels, model_path)
    data = load_data(data_path, n_shapes, n_cons)

    # produced shapes are saved in the results directory
    # see the ground truth in ./results/groundtruth
    ## reconstruct demo shapes
    reconstruct_shapes(sess, ops, data, results_path)

    ## interpolate shapes from data
    interpolate_shapes(sess, ops, data, results_path)

    ## analogy between demo shapes
    analogy(sess, ops, data, results_path)
