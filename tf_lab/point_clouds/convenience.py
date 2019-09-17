'''
Created on September 5, 2017

@author: optas
'''
import numpy as np
from general_tools.simpletons import iterate_in_chunks
import tensorflow as tf
from .. external.structural_pc_losses import losses
nn_distance, approx_match, match_cost = losses()
#from latent_3d_points.external.structural_losses import nn_distance, approx_match, match_cost


def reconstruct_pclouds(autoencoder, pclouds_feed, batch_size, pclouds_gt=None, compute_loss=True):
    recon_data = []
    loss = 0.
    last_examples_loss = 0.     # keep track of loss on last batch which potentially is smaller than batch_size
    n_last = 0.

    n_pclouds = len(pclouds_feed)
    if pclouds_gt is not None:
        if len(pclouds_gt) != n_pclouds:
            raise ValueError()

    n_batches = 0.0
    idx = np.arange(n_pclouds)

    for b in iterate_in_chunks(idx, batch_size):
        feed = pclouds_feed[b]
        if pclouds_gt is not None:
            gt = pclouds_gt[b]
        else:
            gt = None
        rec, loss_batch = autoencoder.reconstruct(feed, GT=gt, compute_loss=compute_loss)
        recon_data.append(rec)
        if compute_loss:
            if len(b) == batch_size:
                loss += loss_batch
            else:  # last index was smaller than batch_size
                last_examples_loss = loss_batch
                n_last = len(b)
        n_batches += 1

    if n_last == 0:
        loss /= n_batches
    else:
        loss = (loss * batch_size) + (last_examples_loss * n_last)
        loss /= ((n_batches - 1) * batch_size + n_last)

    return np.vstack(recon_data), loss


def get_latent_codes(autoencoder, pclouds, batch_size=100):
    latent_codes = []
    idx = np.arange(len(pclouds))
    for b in iterate_in_chunks(idx, batch_size):
        latent_codes.append(autoencoder.transform(pclouds[b]))
    return np.vstack(latent_codes)


def decode_latent_codes(autoencoder, latent_codes, batch_size=100):
    pclouds = []
    idx = np.arange(len(latent_codes))
    for b in iterate_in_chunks(idx, batch_size):
        pclouds.append(autoencoder.decode(latent_codes[b]))
    return np.vstack(pclouds)


def compute_structural_loss(pc1, pc2, batch_size, loss_type):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    n_pc, n_pc_points_1, pc_dim = pc1.shape
    n_pc_, n_pc_points_2, pc_dim_s = pc2.shape

    if n_pc != n_pc_ or pc_dim != pc_dim_s:
        raise ValueError()

    # TF Graph Operations
    pc_1_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points_1, 3))
    pc_2_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points_2, 3))
    if loss_type == 'emd':
        match = approx_match(pc_1_pl, pc_2_pl)
        all_dist_in_batch = match_cost(pc_1_pl, pc_2_pl, match)
    elif loss_type == 'chamfer':
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(pc_1_pl, pc_2_pl)
        all_dist_in_batch = tf.reduce_mean(cost_p1_p2, 1) + tf.reduce_mean(cost_p2_p1, 1)
    else:
        raise ValueError()

    all_dists = []
    for chunk in iterate_in_chunks(np.arange(n_pc), batch_size):
        feed_dict = {pc_1_pl: pc1[chunk], pc_2_pl: pc2[chunk]}
        b = sess.run(all_dist_in_batch, feed_dict=feed_dict)
        all_dists.append(b)
    sess.close()
    return np.array(all_dists)[0]
