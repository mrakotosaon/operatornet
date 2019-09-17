'''
Created on January 13, 2017

@author: optas
'''
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow import random_uniform_initializer, truncated_normal_initializer


def initializer(options, shape=None):
    options = copy.deepcopy(options)
    init_type = options.pop('type')
    if init_type == 'glorot':
        return glorot_initializer(shape[0], shape[1], **options)
    elif init_type == 'truncated_normal':
        return tf.truncated_normal_initializer(**options)
    elif init_type == 'glorot_conv2d':
        return xavier_initializer_conv2d(**options)
    elif init_type == 'uniform':
        if 'minval' not in options:
            options['minval'] = -0.05
        if 'maxval' not in options:
            options['maxval'] = 0.05
        return tf.random_uniform_initializer(**options)
    else:
        raise('Please specify a valid initialization of the variables.')


def glorot_initializer(fan_in, fan_out, constant=1.0, uniform=True, dtype=tf.float32):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    SEE: https://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
    '''
    with tf.device('/cpu:0'):
        if uniform:
            init_range = constant * np.sqrt(6.0 / (fan_in + fan_out))
            return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)
        else:
            stddev = constant * np.sqrt(2.0 / (fan_in + fan_out))
            return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def orthogonal_initializer(shape, scale=1.1):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    SEE: https://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]
