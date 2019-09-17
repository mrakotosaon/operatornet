'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only, encoder_with_convs_and_symmetry_new


def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    encoder = encoder_with_convs_and_symmetry_new
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args


def conv_architecture_ala_nips_17(n_pc_points):
    if n_pc_points == 2048:
        encoder_args = {'n_filters': [128, 128, 256, 512],
                        'filter_sizes': [40, 20, 10, 10],
                        'strides': [1, 2, 2, 1]
                        }
    else:
        assert(False)

    n_input = [n_pc_points, 3]

    decoder_args = {'layer_sizes': [1024, 2048, np.prod(n_input)]}

    res = {'encoder': encoder_with_convs_and_symmetry,
           'decoder': decoder_with_fc_only,
           'encoder_args': encoder_args,
           'decoder_args': decoder_args
           }
    return res


def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 500,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return params
