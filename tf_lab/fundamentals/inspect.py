'''
Created on Jan 9, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
from . utils import format_scope_name, count_cmp_to_value

SUMMARIES_COLLECTION = 'summary_tags'       # Keeping all summaries in this collection, each summary being stored as value of a dictionary.
SUMMARY_TAG = 'tag'                         # Used as the key on a dictionary storing the name of the summary operation.
SUMMARY_TENSOR = 'tensor'                   # Used as the key on a dictionary storing the tensor of the summary operation.


SUMMARIES_COLLECTION = 'summary_tags'    # Keeping all summaries in this collection, each summary being stored as part of a dictionary.
SUMMARY_TAG = 'tag'                         # Used as the key on a dictionary storing the name of the summary operation.
SUMMARY_TENSOR = 'tensor'                   # Used as the key on a dictionary storing the tensor of the summary operation.

# Fix for TF 0.12
try:
    tf012 = True
    merge_summary = tf.summary.merge
except Exception:
    tf012 = False
    merge_summary = tf.merge_summary


def summarize_gradients(grads_and_var, summary_collection, prefix='', postfix=''):
    """ summarize_gradients.
    Arguments:
        grads: list of `Tensor`. The gradients to monitor.
        summary_collection: A collection to add this summary to and
            also used for returning a merged summary over all its elements.
    Returns:
        `Tensor`. Merge of all summary in 'summary_collection'
    """
    grad_sum = add_gradients_summary(grads_and_var, prefix, postfix, summary_collection)
    return merge_summary(grad_sum)


def get_summary_if_exists(tag):
    """ summary_exists.
    Retrieve a summary exists if exists, or None.
    Arguments:
        tag: `str`. The summary name.
    """
    return next((item[SUMMARY_TENSOR] for item in tf.get_collection(SUMMARIES_COLLECTION) if item[SUMMARY_TAG] == tag), None)


def get_summary(stype, tag, value=None, collection_key=None, break_if_exists=False):
    """ get_summary.

    Create or retrieve a summary. It keep tracks of all graph summaries
    through summary_tags collection. If a summary tags already exists,
    it will return that summary tensor or raise an error (according to
    'break_if_exists').
    Arguments:
        stype: `str`. Summary type: 'histogram', 'scalar' or 'image'.
        tag: `str`. The summary tag (name).
        value: `Tensor`. The summary initialization value. Default: None.
        collection_key: `str`. If specified, the created summary will be
            added to that collection (optional).
        break_if_exists: `bool`. If True, if a summary with same tag already
            exists, it will raise an exception (instead of returning that
            existing summary).
    Returns:
        The summary `Tensor`.
    """
    summ = get_summary_if_exists(tag)

    if summ is None:
        if value is None:
            raise Exception("Summary doesn't exist, a value must be "
                            "specified to initialize it.")
        if stype == "histogram":
            summ = tf.summary.histogram(tag, value)
        elif stype == "scalar":
            summ = tf.summary.scalar(tag, value)
        elif stype == "image":
            summ = tf.summary.image(tag, value)
        else:
            raise ValueError("Unknown summary type: '" + str(stype) + "'")

        tf.add_to_collection(SUMMARIES_COLLECTION, {SUMMARY_TAG: tag, SUMMARY_TENSOR: summ})

    elif break_if_exists:
        raise ValueError("Error: Summary tag already exists! (to ignore this "
                         "error, set add_summary() parameter 'break_if_exists'"
                         " to False)")
    else:
        summ = summ[SUMMARY_TENSOR]

    if collection_key is not None:
            tf.add_to_collection(collection_key, summ)

    return summ


def add_gradients_summary(grad_and_vars, collection_key, name_prefix="", name_suffix=""):
    """ add_gradients_summary.
    Add histogram summary for given gradients.
    Arguments:
        grads: A list of `Tensor`. The gradients to summarize.
        name_prefix: `str`. A prefix to add to summary scope.
        name_suffix: `str`. A suffix to add to summary scope.
        collection_key: `str`. A collection to store the summaries.
    Returns:
        The list of created gradient summaries.
    """

    # Add histograms for gradients.
    summ = []
    for grad, var in grad_and_vars:
        if grad is not None:
            summ_name = format_scope_name(var.op.name, name_prefix, "Gradients/" + name_suffix)
            summ.append(get_summary("histogram", summ_name, grad, collection_key))
    return summ


def fraction_of_grads_less_than(grads_and_vars, bound, tag, collections=None):
    counter = []
    sizes = []
    for grad, _ in grads_and_vars:
        if grad is not None:
            counter.append(count_cmp_to_value(grad, bound, tf.less_equal))
            sizes.append(tf.cast(tf.size(grad), dtype=tf.float32))

    res_val = tf.add_n(counter) / tf.add_n(sizes)
    return tf.scalar_summary(tag, res_val, collections)


def trainable_variables(in_graph=None):
    if in_graph is None:
        return tf.trainable_variables()
    else:
        return in_graph.get_collection('trainable_variables')


def count_trainable_parameters(in_graph=None, name_space=None):
    if in_graph is None:
        in_graph = tf.get_default_graph()

    total_parameters = 0
    # for variable in tf.trainable_variables():
    for variable in in_graph.get_collection('trainable_variables'):
        if name_space is not None and name_space not in variable.name:
            continue
        # shape is an array of tf.Dimension
        shape = variable.get_shape().as_list()
        variable_parametes = np.prod(shape)
        total_parameters += variable_parametes
    return total_parameters


def sparsity_summary_of_trainable():
    summaries = []
    with tf.device('/cpu:0'):
        for var in tf.trainable_variables():
            summaries.append(tf.scalar_summary('sparsity_' + var.op.name, tf.nn.zero_fraction(var)))
    return summaries
