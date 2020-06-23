import os
from functools import reduce
import torch
import glob
import math
import numpy as np


class Config(object):

    def __init__(self):

        self.dataset = ['ptb', 'bc', 'yelp', 'yahoo'][2]
        self.model = ['lm',
                      'dae',
                      'vae',
                      'vae-couple',
                      'vae-nf',
                      'vae-nf-couple',
                      'wae',
                      'wae-couple',
                      'wae-nf',
                      'wae-nf-couple',
                      'beta-vae',
                      'beta-vae-couple',
                      's-vae',
                      's-vae-couple',
                      'cnn-vae',
                      'cnn-vae-couple',
                      'cyc-anneal-vae',
                      'cyc-anneal-vae-couple',
                      'surprising-fix',
                      'surprising-fix-couple',
                      ][0]
        self.train_mode = ['gen'][0]
        self.use_lstm = False

        # Training configuration.
        self.best_metric = None
        self.num_iters = 60000
        self.start_decay = 30000
        self.dropout = 0.2
        self.weight_decay = 0
        self.batch_size = 32
        self.gen_lr = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.clip_norm = 30.0
        self.clip_value = float('inf')

        # Data configuration.
        self.max_len = 200

        # Test configuration.
        self.beam_size = 1

        if self.dataset == 'ptb':
            # Model configuration.
            self.hid_dim = 128
            self.emb_dim = 200
            self.enc_layers = 1
            self.dec_layers = 1
            assert self.enc_layers == self.dec_layers
            self.pre_emb = False
            self.latent_dim = 32
            self.WEAtt_type = 'none'

            if self.model == 'lm':
                pass
            elif self.model == 'dae':
                pass
            elif self.model == 'vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
            elif self.model == 'vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = [0.1, 10.0][0]
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
            elif self.model == 'beta-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
            elif self.model == 'beta-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
                self.lambda_couple = 0.1
            elif self.model == 'vae-nf':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'vae-nf-couple':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = [0.0, 0.1][1]
            elif self.model == 'wae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
            elif self.model == 'wae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.lambda_couple = 0.1
            elif self.model == 'wae-nf':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'wae-nf-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = 0.1
            elif self.model == 's-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
            elif self.model == 's-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_couple = [0, 7.5][1]
            elif self.model == 'cnn-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = ['small', 'medium', 'large'][1]
                self.n_filter = 256
            elif self.model == 'cnn-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = ['small', 'medium', 'large'][1]
                self.n_filter = 256
                self.lambda_couple = 0
            elif self.model == 'cyc-anneal-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.M = 4
                self.R = 0.5
            elif self.model == 'cyc-anneal-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 0.1
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.M = 4
                self.R = 0.5
            elif self.model == 'surprising-fix':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_fb = 6
            elif self.model == 'surprising-fix-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 0.1
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.lambda_fb = 6
            else:
                raise ValueError()
        elif self.dataset == 'bc':
            pass
        elif self.dataset == 'yelp':
            # Model configuration.
            self.hid_dim = 256
            self.emb_dim = 200
            self.enc_layers = 1
            self.dec_layers = 1
            assert self.enc_layers == self.dec_layers
            self.pre_emb = False
            self.latent_dim = 32
            self.WEAtt_type = 'none'

            if self.model == 'lm':
                pass
            elif self.model == 'dae':
                pass
            elif self.model == 'vae':
                self.kl_annealing = True
                self.annealing_start = 1000  # When set as 2000, KL does not converge (scale: e+23), which is weird.
                self.annealing_step = 40000
                self.lambda_BoW = 0
            elif self.model == 'vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000  # Following vae
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = [0.1, 10.0][0]
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
            elif self.model == 'beta-vae':
                self.kl_annealing = True
                self.annealing_start = 1000  # When set as 2000, KL does not converge (scale: e+23), which is weird.
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
            elif self.model == 'beta-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000  # Following beta-vae
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
                self.lambda_couple = 0.1
            elif self.model == 'vae-nf':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'vae-nf-couple':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = [0.0, 0.1][1]
            elif self.model == 'wae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
            elif self.model == 'wae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.lambda_couple = 0.1
            elif self.model == 'wae-nf':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'wae-nf-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = 0.1
            elif self.model == 's-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
            elif self.model == 's-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_couple = [0, 7][1]
            elif self.model == 'cnn-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = ['small', 'medium', 'large'][2]
                self.n_filter = 128
            elif self.model == 'cnn-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = ['small', 'medium', 'large'][2]
                self.n_filter = 128
                self.lambda_couple = 0.1
            elif self.model == 'cyc-anneal-vae':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.M = 4
                self.R = 0.5
            elif self.model == 'cyc-anneal-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 1.0
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.M = 4
                self.R = 0.5
            elif self.model == 'surprising-fix':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_fb = 6
            elif self.model == 'surprising-fix-couple':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 0.1
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.lambda_fb = 6
            else:
                raise ValueError()
        elif self.dataset == 'yahoo':
            # Model configuration.
            self.hid_dim = 256
            self.emb_dim = 200
            self.enc_layers = 1
            self.dec_layers = 1
            assert self.enc_layers == self.dec_layers
            self.pre_emb = False
            self.latent_dim = 32
            self.WEAtt_type = 'none'

            if self.model == 'lm':
                pass
            elif self.model == 'dae':
                pass
            elif self.model == 'vae':
                self.kl_annealing = True
                self.annealing_start = 1000  # When set as 2000, KL does not converge (scale: e+23), which is weird.
                self.annealing_step = 40000
                self.lambda_BoW = 0
            elif self.model == 'vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000  # Following vae.
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = [0.1, 10.0][0]
                self.euclidean_dist = [False, True][1]
            elif self.model == 'beta-vae':
                self.kl_annealing = True
                self.annealing_start = 1000  # When set as 2000, KL does not converge (scale: e+23), which is weird.
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
            elif self.model == 'beta-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000  # Following beta-vae
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.beta = [0.8, 1.2, 1.4][0]
                self.lambda_couple = 0.1
            elif self.model == 'vae-nf':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'vae-nf-couple':
                self.cond_params = False
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = 0.1
            elif self.model == 'wae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
            elif self.model == 'wae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.lambda_couple = 0.1
            elif self.model == 'wae-nf':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
            elif self.model == 'wae-nf-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.cond_params = False
                self.lambda_mmd = 10
                self.mmd_kernel = ['rbf', 'imq'][1]
                self.n_flows = 3
                self.flow_type = ['planar'][0]
                self.lambda_couple = 0.1
            elif self.model == 's-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
            elif self.model == 's-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_couple = 5
            elif self.model == 'cnn-vae':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = 'large'
                self.n_filter = 256
            elif self.model == 'cnn-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.scale = 'large'
                self.n_filter = 256
                self.lambda_couple = 0.1
            elif self.model == 'cyc-anneal-vae':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.M = 4
                self.R = 0.5
            elif self.model == 'cyc-anneal-vae-couple':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 1.0
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.M = 4
                self.R = 0.5
            elif self.model == 'surprising-fix':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_fb = 6
            elif self.model == 'surprising-fix-couple':
                self.kl_annealing = True
                self.annealing_start = 1000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = 0.1
                self.lambda_seq_couple = 1
                self.euclidean_dist = [False, True][0]
                self.lambda_fb = 6
            else:
                raise ValueError()
        else:
            raise ValueError()

        # Miscellaneous.
        self.num_workers = 8
        self.use_tensorboard = True
        self.ROUND = 4
        self.seed = 0
        self.gpu = torch.cuda.is_available()

        # Step size.
        self.log_step = 10
        self.sample_step = 1000
        self.lr_decay_step = 2000

        # Directories.
        self.log_dir = self.model_specific_dir('outputs/logs')
        remove_all_under(self.log_dir)
        self.save_model_dir = self.model_specific_dir('outputs/saved_model')
        self.sample_dir = self.model_specific_dir('outputs/sampled_results')
        self.tmp_dir = self.model_specific_dir('outputs/temp_results')

    def model_specific_dir(self, root):
        """ model-normalization """

        name_components = [
            self.dataset,
            self.model,
        ]
        if self.weight_decay > 0:
            name_components.append('wd{}'.format(self.weight_decay))

        if self.model == 'lm':
            pass
        elif self.model == 'dae':
            pass
        elif self.model == 'vae':
            pass
        elif self.model == 'vae-couple':
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
            if self.euclidean_dist:
                name_components.append('euclidean')
            if self.lambda_seq_couple != 1:
                name_components.append('lambda_seq_couple{}'.format(self.lambda_seq_couple))
        elif self.model == 'beta-vae':
            name_components.append('beta{}'.format(self.beta))
        elif self.model == 'beta-vae-couple':
            name_components.append('beta{}'.format(self.beta))
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
        elif self.model == 'vae-nf':
            if not self.cond_params:
                name_components.append('noncond')
        elif self.model == 'vae-nf-couple':
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
            if not self.cond_params:
                name_components.append('noncond')
        elif self.model == 'wae':
            pass
        elif self.model == 'wae-couple':
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
        elif self.model == 'wae-nf':
            if not self.cond_params:
                name_components.append('noncond')
        elif self.model == 'wae-nf-couple':
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
            if not self.cond_params:
                name_components.append('noncond')
        elif self.model == 's-vae':
            pass
        elif self.model == 's-vae-couple':
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
        elif self.model == 'cnn-vae':
            name_components.append(self.scale)
        elif self.model == 'cnn-vae-couple':
            name_components.append(self.scale)
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
        elif self.model == 'cyc-anneal-vae':
            name_components.append('M{}'.format(self.M))
            name_components.append('R{}'.format(self.R))
        elif self.model == 'cyc-anneal-vae-couple':
            name_components.append('M{}'.format(self.M))
            name_components.append('R{}'.format(self.R))
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
            if self.euclidean_dist:
                name_components.append('euclidean')
            if self.lambda_seq_couple != 1:
                name_components.append('lambda_seq_couple{}'.format(self.lambda_seq_couple))
        elif self.model == 'surprising-fix':
            name_components.append('lambda_fb{}'.format(self.lambda_fb))
        elif self.model == 'surprising-fix-couple':
            name_components.append('lambda_fb{}'.format(self.lambda_fb))
            name_components.append('lambda_couple{}'.format(self.lambda_couple))
            if self.euclidean_dist:
                name_components.append('euclidean')
            if self.lambda_seq_couple != 1:
                name_components.append('lambda_seq_couple{}'.format(self.lambda_seq_couple))
        else:
            raise ValueError()
        ret = os.path.join(root, '-'.join(name_components))
        if not os.path.exists(ret):
            os.mkdir(ret)
        return ret


def remove_all_under(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)
