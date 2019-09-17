'''
Created on Apr 27, 2017

@author: optas
'''

import tensorflow as tf


def reset_tf_graph():
    ''' Reset's all variables of default-tf graph. Useful for jupyter.
    '''
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
