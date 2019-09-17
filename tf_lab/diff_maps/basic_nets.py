'''
Created on Jan 9, 2018

@author: optas
'''

from collections import defaultdict
import tensorflow as tf
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d

from tf_lab import Neural_Net
from tf_lab.point_clouds.encoders_decoders import encoder_with_convs_and_symmetry_new, decoder_with_fc_only
from tf_lab.fundamentals.inspect import count_trainable_parameters
from tflearn import is_training

from .in_out import classes_of_tasks

n_best_pc_parms = 34124


def pc_net(n_pc_points, task, n_filters, n_neurons, verbose=False):
    n_classes = classes_of_tasks(task)
    labels_pl, last_nn = start_end_of_nets(task)
    with tf.variable_scope('pc_based_net'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, 3))
        layer = encoder_with_convs_and_symmetry_new(feed_pl, n_filters=n_filters, b_norm=False)
        n_neurons = n_neurons + [n_classes]
        net_out = decoder_with_fc_only(layer, n_neurons, b_norm=False)
        if last_nn == 'relu':
            net_out = tf.nn.relu(net_out)

        if verbose:
            n_tp = count_trainable_parameters()
            print '#PARAMS ', n_tp

    return net_out, feed_pl, labels_pl


def diff_mlp_net(n_cons, task, verbose=False):
    n_classes = classes_of_tasks(task)
    labels_pl, last_nn = start_end_of_nets(task)
    f_layer = mlp_neurons_on_first_layer(n_cons)
    with tf.variable_scope('mlp_diff_based_net'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
        layer = fully_connected(feed_pl, f_layer, activation='relu', weights_init='xavier')
        layer = fully_connected(layer, 50, activation='relu', weights_init='xavier')
        layer = fully_connected(layer, 100, activation='relu', weights_init='xavier')
        net_out = fully_connected(layer, n_classes, activation=last_nn, weights_init='xavier')
        n_tp = count_trainable_parameters()
        if verbose:
            print '#PARAMS ', n_tp

        assert (n_tp <= 0.01 * n_best_pc_parms + n_best_pc_parms)
        assert (n_tp >= n_best_pc_parms - 0.01 * n_best_pc_parms)
    return net_out, feed_pl, labels_pl

                                    

def diff_conv_net(n_cons, task, verbose=False):
    n_classes = classes_of_tasks(task)
    labels_pl, last_nn = start_end_of_nets(task)
    with tf.variable_scope('conv_diff_based_net'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
        layer = tf.expand_dims(feed_pl, -1)
        
        if n_cons == 20:
            layer = conv_2d(layer, nb_filter=10, filter_size=2, strides=1, activation='relu')
        elif n_cons == 40:
            if task == "pose_clf":
                layer = conv_2d(layer, nb_filter=10, filter_size=6, strides=2, activation='relu')
            else:
                layer = conv_2d(layer, nb_filter=10, filter_size=3, strides=2, activation='relu')
        elif n_cons == 50:
            layer = conv_2d(layer, nb_filter=10, filter_size=4, strides=2, activation='relu')            
        else:
            assert(False)            
        layer = conv_2d(layer, nb_filter=10, filter_size=4, strides=2, activation='relu')
        net_out = fully_connected(layer, n_classes, activation=last_nn, weights_init='xavier')
        
    n_tp = count_trainable_parameters()
    if verbose:
        print '#PARAMS ', n_tp
        
    return net_out, feed_pl, labels_pl


class Basic_Net(Neural_Net):
    def __init__(self, net_out, feed_pl, label_pl, name='todo', graph=None):
        Neural_Net.__init__(self, name, graph)
        self.net_out = net_out
        self.feed_pl = feed_pl
        self.labels_pl = label_pl

    def define_loss(self, task):
        n_classes = classes_of_tasks(task)
        if task == 'regression':
            self.loss = tf.losses.mean_squared_error(self.labels_pl, self.net_out)
        else:
            self.prediction = tf.argmax(self.net_out, axis=1)
            correct_pred = tf.equal(self.prediction, self.labels_pl)
            self.avg_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            one_hot_labels = tf.one_hot(self.labels_pl, depth=n_classes)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.net_out, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy)

    def define_opt(self, learning_rate):
        opt = tf.train.AdamOptimizer(learning_rate)
        self.opt_step = opt.minimize(self.loss)

    def train(self, n_epochs, batch_size, dataset, task, verbose=False):
        stats = defaultdict(list)
        train_data = dataset['train']
        batches_for_epoch = train_data.n_examples / batch_size
        for _ in range(n_epochs):
            is_training(True, session=self.sess)
            for _ in range(batches_for_epoch):
                batch_d, batch_l, _ = train_data.next_batch(batch_size)
                feed_dict = {self.feed_pl: batch_d, self.labels_pl: batch_l}
                self.sess.run([self.opt_step], feed_dict=feed_dict)
            epoch = self.sess.run(self.epoch.assign_add(tf.constant(1.0)))
            is_training(False, session=self.sess)

            if verbose:
                print epoch,
            for s in ['train', 'test', 'val']:
                feed_dict = {self.feed_pl: dataset[s].feed, self.labels_pl: dataset[s].labels}
                if task == 'regression':
                    r = self.sess.run([self.loss], feed_dict=feed_dict)
                else:
                    r = self.sess.run([self.avg_accuracy], feed_dict=feed_dict)
                stats[s].append(r)
                if verbose:
                    print r,
            if verbose:
                print

        return stats


def start_end_of_nets(task):
    n_classes = classes_of_tasks(task)
    if task == 'regression':
        labels_pl = tf.placeholder(tf.float32, shape=[None, n_classes])
        last_nn = 'relu'
    else:
        labels_pl = tf.placeholder(tf.int64, shape=[None])
        last_nn = 'linear'
    return labels_pl, last_nn


def pc_versions(ver):
    if ver == 'v1':
        n_filters = [32, 64, 64]
        n_neurons = [64]
    elif ver == 'v2':
        n_filters = [64, 128, 128]
        n_neurons = [64]
    elif ver == 'v3':
        n_filters = [64, 128, 128]
        n_neurons = [64, 128]
    else:
        assert(False)
    return n_filters, n_neurons


def mlp_neurons_on_first_layer(n_cons):
    if n_cons == 5:
        f_layer = 369
    elif n_cons == 10:
        f_layer = 185
    elif n_cons == 20:
        f_layer = 62
    elif n_cons == 30:
        f_layer = 29
    elif n_cons == 40:
        f_layer = 17
    elif n_cons == 50:
        f_layer = 11
    return f_layer