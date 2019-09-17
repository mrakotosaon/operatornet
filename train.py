from general_tools.notebook.gpu_utils import setup_one_gpu
setup_one_gpu(1)
import tensorflow as tf
import numpy as np
import os
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d
from general_tools.notebook.tf import reset_tf_graph
from tf_lab.data_sets.numpy_dataset import NumpyDataset
from scipy.io import loadmat

def load_SD(path, n_shapes, n_cons):
    diff_maps = loadmat(path)['diff_maps']
    diff_maps = diff_maps.reshape(n_shapes, n_cons, n_cons)
    diff_maps =np.expand_dims(diff_maps, -1)
    return diff_maps

def load_shape_diff(data_path,n_shapes, n_cons):
    # Load shape differences
    diff_area_maps_path = os.path.join(data_path,"Area.mat")
    diff_area_maps =load_SD(diff_area_maps_path, n_shapes, n_cons)
    diff_ext_path = os.path.join(data_path, "Ext.mat")
    diff_ext_maps =load_SD(diff_ext_path,n_shapes, n_cons)
    diff_conf_path = os.path.join(data_path, "Conf.mat")
    diff_conf_maps =load_SD(diff_conf_path,n_shapes, n_cons)
    return diff_ext_maps, diff_area_maps, diff_conf_maps

def get_data(channels, data_path,n_shapes, n_cons, n_pc_points,testset_ratio = 0.2):
    # build dataset
    # WARNING: note that test and train sets are buid randomly here contrary to the method we use in the paper
    diff_ext_maps, diff_area_maps, diff_conf_maps = load_shape_diff(data_path,n_shapes, n_cons)
    if channels == "A+C+E":
        diff_maps = np.concatenate([diff_ext_maps, diff_area_maps, diff_conf_maps], 3)
    else:
        raise NotImplementedError

    # Load point clouds
    label_path = os.path.join(data_path, "Label.mat")
    label =loadmat(label_path)['P']
    label = label.reshape(n_shapes, n_pc_points,  3)
    n_test = int(n_shapes*testset_ratio)
    seed = 42
    np.random.seed(seed)
    perm = np.random.permutation(n_shapes)
    diff_maps_trainingset = diff_maps[perm][:-n_test]
    diff_maps_testset = diff_maps[perm][-n_test:]
    label_testset = label[perm][-n_test:]
    label_trainingset = label[perm][:-n_test]

    net_data = dict()

    net_data['train'] = NumpyDataset([diff_maps_trainingset, label_trainingset])
    net_data['test'] = NumpyDataset([diff_maps_testset, label_testset])
    return net_data


def diff_reconstructor(n_cons, n_pc_points, n_channels):
    with tf.variable_scope('conv_based_reconstructor'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons, n_channels))
        labels_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, 3))
        # encoder
        layer = conv_2d(feed_pl, nb_filter=8, filter_size=3, strides=2, activation='relu')
        # decoder
        layer = fully_connected(layer, 1024, activation="relu")
        layer = fully_connected(layer, 1024, activation="relu")
        net_out = fully_connected(layer,n_pc_points * 3 )
        net_out = tf.reshape(net_out, [-1, n_pc_points, 3])
        return net_out, feed_pl, labels_pl

def align(X, Y):
    # align shapes from X to optimal tranformation between X and Y
    n_pc_points = X.shape[1]
    mu_x = tf.reduce_mean(X, axis = 1)
    mu_y =  tf.reduce_mean(Y, axis = 1)

    concat_mu_x = tf.tile(tf.expand_dims(mu_x,1), [1, n_pc_points, 1])
    concat_mu_y = tf.tile(tf.expand_dims(mu_y,1), [1, n_pc_points, 1])

    centered_y = tf.expand_dims(Y - concat_mu_y, 2)
    centered_x = tf.expand_dims(X - concat_mu_x, 2)

    # transpose y
    centered_y = tf.einsum('ijkl->ijlk', centered_y)

    mult_xy = tf.einsum('abij,abjk->abik', centered_y, centered_x)
    # sum
    C = tf.einsum('abij->aij', mult_xy)
    s, u,v = tf.svd(C)
    v = tf.einsum("aij->aji", v)

    R_opt = tf.einsum("aij,ajk->aik", u, v)
    t_opt = mu_y - tf.einsum("aki,ai->ak", R_opt, mu_x)
    concat_R_opt = tf.tile(tf.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    concat_t_opt = tf.tile(tf.expand_dims(t_opt,1), [1, n_pc_points, 1])
    opt_labels =  tf.einsum("abki,abi->abk", concat_R_opt, X) + concat_t_opt
    return opt_labels

def train_one_epoch(batches_for_epoch, net_data, batch_size, feed_pl, labels_pl, train_step, loss, sess):
    epoch_loss = []
    for _ in range(batches_for_epoch):
        feed, gt = net_data['train'].next_batch(batch_size)
        feed_dict = {feed_pl:feed, labels_pl:gt}
        _, l= sess.run([train_step, loss], feed_dict)
        epoch_loss.append(l)
    return np.mean(epoch_loss)

def test_one_epoch(net_data, batch_size, feed_pl, labels_pl, loss, sess, n_test_batches_for_epoch):
    t_err = []
    for test_batch_idx in range(n_test_batches_for_epoch):
        feed, gt = net_data['test'].next_batch(batch_size)
        feed_dict = {feed_pl:feed, labels_pl:gt}
        t = sess.run([loss], feed_dict)
        t_err.append(t)
    perf = np.mean(t_err)
    return perf

def train():
    learning_rate = 0.0001
    max_epochs = 450
    data_path= "Data/demo_dfaust_data"
    n_pc_points = 1000
    n_cons = 60 # size of reduced basis
    channels = "A+C+E"
    n_channels = 3
    batch_size = 32
    n_shapes = 10240  # total number of shapes inside dataset
    model_path =  "models/best_model_{channels}.ckpt".format(channels = channels)
    n_plot = 2 # show training losses every n_plot epochs

    net_data = get_data(channels, data_path, n_shapes, n_cons, n_pc_points)

    # init graph
    reset_tf_graph()
    x_reconstr, feed_pl, labels_pl = diff_reconstructor(n_cons, n_pc_points, n_channels)
    loss = tf.losses.mean_squared_error(align(labels_pl, x_reconstr), x_reconstr)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver()

    # train network
    train_errors = [] # these lists contain the training curves and can be displayed
    test_errors = []
    n_batches_for_epoch = net_data['train'].n_examples / batch_size
    n_test_batches_for_epoch = net_data['test'].n_examples / batch_size

    best = np.inf
    for epoch in range(max_epochs):
        period_loss = 0.0
        period_loss += train_one_epoch(n_batches_for_epoch, net_data, batch_size,feed_pl,
                                                           labels_pl, train_step, loss, sess)
        if epoch % n_plot == 0:
            print("epoch : {}/{} loss: {}".format(epoch, max_epochs, period_loss/n_plot))
            test_loss = test_one_epoch(net_data, batch_size,feed_pl, labels_pl, loss, sess, n_test_batches_for_epoch)
            test_errors.append(test_loss)
            if test_loss<best:
                best = test_loss
                save_path = saver.save(sess, model_path)
                print("saved best model to : {}".format(save_path))
            print("test loss: {}".format(test_loss))
            train_errors.append(period_loss / float(n_plot))


if __name__ == '__main__':
    train()
