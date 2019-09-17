'''
Created on December 26, 2016

@author: optas
'''

import itertools
import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(5, 5),
                          save_file=None, plt_nums=False):
    '''This function prints and plots the confusion matrix.'''
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=80)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if plt_nums:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)


def _scale_2d_embedding(two_dim_emb):
    two_dim_emb -= np.min(two_dim_emb, axis=0)  # scale x-y in [0,1]
    two_dim_emb /= np.max(two_dim_emb, axis=0)
    return two_dim_emb


def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None):
    '''
    Input:
        two_dim_emb: (N x 2) numpy array: arbitrary 2-D embedding of data.
        image_files: (list) of strings pointing to images. Specifically image_files[i] should be an image associated with
                     the datum whose coordinates are given in two_dim_emb[i].
        big_dim:     (int) height of output 'big' grid rectangular image.
        small_dim:   (int) height to which each individual rectangular image/thumbnail will be resized.
    '''
    ceil = np.ceil
    mod = np.mod
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.zeros((big_dim, big_dim, 3), dtype='uint8')
#     out_image = np.ones((big_dim, big_dim, 3), dtype='uint8') * 255

    for i, im_file in enumerate(image_files):
        #  Determine location on grid
        a = ceil(x[i, 0] * (big_dim - small_dim) + 1)
        b = ceil(x[i, 1] * (big_dim - small_dim) + 1)
        a = int(a - mod(a - 1, small_dim) + 1)
        b = int(b - mod(b - 1, small_dim) + 1)

        if out_image[a, b, 0] != 0:
#         if out_image[a, b, 0] != 255:
            continue    # Spot already filled.

        fig = cv2.imread(im_file)
#         fig = read_transparent_png(im_file)
        fig = cv2.resize(fig, (small_dim, small_dim))

        try:
            out_image[a:a + small_dim, b:b + small_dim, :] = fig
        except:
                print 'the code here fails. fix it.'
                print a
        continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image


def plot_2d_embedding_in_grid_forceful(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None):
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.zeros((big_dim, big_dim, 3), dtype='uint8')
    N = two_dim_emb.shape[0]
    xnum = int(big_dim / float(small_dim))
    ynum = int(big_dim / float(small_dim))
    free = np.ones(N, dtype=np.bool)

    grid_2_img = np.ones((xnum, ynum), dtype='int') * -1
    res = float(small_dim) / float(big_dim)
    for i in xrange(xnum):
        for j in xrange(ynum):
            sorted_indices = np.argsort((x[:, 0] - i * res)**2 + (x[:, 1] - j * res)**2)
            possible = sorted_indices[free[sorted_indices]]

            if len(possible) > 0:
                picked = possible[0]
                free[picked] = False
                grid_2_img[i, j] = picked
            else:
                break

    for i in xrange(xnum):
        for j in xrange(ynum):
            if grid_2_img[i, j] > -1:
                im_file = image_files[grid_2_img[i, j]]
                fig = cv2.imread(im_file)
                fig = cv2.resize(fig, (small_dim, small_dim))
                try:
                    out_image[i * small_dim:(i + 1) * small_dim, j * small_dim:(j + 1) * small_dim, :] = fig
                except:
                    print 'the code here fails. fix it.'
                    print im_file
                continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)

    return out_image
