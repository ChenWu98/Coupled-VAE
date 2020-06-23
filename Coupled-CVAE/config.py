import os
from functools import reduce
import torch
import glob
import math
import numpy as np


class Config(object):

    def __init__(self):

        self.dataset = ['switchboard'][0]  # TODO
        self.model = ['seq2seq',  # 0
                      'cvae',  # 1
                      'cvae-couple',  # 2
                      ][2]  # TODO
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

        if self.dataset == 'switchboard':
            # Model configuration.
            self.hid_dim = 128
            self.emb_dim = 200

            self.enc_layers = 1
            self.dec_layers = 1
            assert self.enc_layers == self.dec_layers
            self.pre_emb = False
            self.latent_dim = 32
            self.WEAtt_type = 'none'

            if self.model == 'seq2seq':
                pass
            elif self.model == 'cvae':
                self.kl_annealing = [False, True][1]
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = [0, 1][0]
            elif self.model == 'cvae-couple':
                self.kl_annealing = [False, True][1]
                self.annealing_start = 2000
                self.annealing_step = 40000
                self.lambda_BoW = 0
                self.lambda_couple = [0.1, 0.5, 1, 2][3]  # TODO
                self.lambda_seq_couple = 1  # TODO
                self.euclidean_dist = [False, True][0]  # TODO
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

        if self.model == 'seq2seq':
            pass
        elif self.model == 'cvae':
            pass
        elif self.model == 'cvae-couple':
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
