'''
Created on February 4, 2017

@author: optas

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d, highway_conv_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected
#from . spatial_transformer import transformer as pcloud_spn
#from . point_net_pp.modules import pointnet_pp_module
from .. fundamentals.layers import conv_1d_tranpose
from .. fundamentals.utils import expand_scope_by_name, replicate_parameter_for_all_layers

dropout = tf.nn.dropout
# from tflearn.layers.core import fully_connected, dropout


def encoder_with_convs_and_symmetry_new(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
#     dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    if spn:
        transformer = pcloud_spn(in_signal)
        in_signal = tf.batch_matmul(in_signal, transformer)
        print 'Spatial transformer was activated.'

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] != 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print layer

    if closing is not None:
        layer = closing(layer)
        print layer

    return layer


def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                    b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                    symmetry=tf.reduce_max, dropout_prob=None, scope=None, reuse=False):

    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    warnings.warn('Using old architecture.')
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    if spn:
        transformer = pcloud_spn(in_signal)
        in_signal = tf.batch_matmul(in_signal, transformer)
        print 'Spatial transformer was activated.'

    name = 'encoder_conv_layer_0'
    scope_i = expand_scope_by_name(scope, name)
    layer = conv_1d(in_signal, nb_filter=n_filters[0], filter_size=filter_sizes[0], strides=strides[0], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

    if b_norm:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

    layer = non_linearity(layer)

    if dropout_prob is not None and dropout_prob[0] > 0:
        layer = dropout(layer, 1.0 - dropout_prob[0])

    for i in xrange(1, n_layers):
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

        if b_norm:
            name += '_bnorm'
            #scope_i = expand_scope_by_name(scope, name) # FORGOT TO PUT IT BEFORE ICLR
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

    if symmetry is not None:
        layer = symmetry(layer, axis=1)

    return layer

def encoder_with_grouping_and_interpolation(in_signal, grp_config=None, interp_config=None, b_norm=True, bn_decay=None, use_normal=False, scope=None, reuse=False, is_training=None):            

    sa_fp=pointnet_pp_module(in_signal, grp_config.filters, grp_config.points, grp_config.radii, grp_config.samples, interp_config.filters, interp_config.idx, interp_config.use_pts0, return_all=interp_config.return_all, b_norm=b_norm, bn_decay=bn_decay, use_normal=use_normal, scope=scope, reuse=reuse, is_training=is_training)
    #note, if no interp layers, we need to squeeze the pointwise features (1,n_dim) to (n_dim)
    sa_fp=tf.reshape(sa_fp,[-1,sa_fp.get_shape()[-1]])
    return sa_fp


def encoder_with_covns_and_grouping(reuse=False, scope=None):
    '''
        Point-Net++ encoder.
    '''
    pass


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer


def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d_tranpose, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print 'Building Decoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer
