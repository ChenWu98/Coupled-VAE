import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn.utils import clip_grad_norm_
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from Modules.lm import LM
from Modules.dae import DAE
from Modules.vae import VAE
from Modules.vae_couple import VAE_COUPLE
from Modules.vae_nf import VAE_NF
from Modules.vae_nf_couple import VAE_NF_COUPLE
from Modules.wae import WAE
from Modules.wae_couple import WAE_COUPLE
from Modules.wae_nf import WAE_NF
from Modules.wae_nf_couple import WAE_NF_COUPLE
from Modules.s_vae import S_VAE
from Modules.s_vae_couple import S_VAE_COUPLE
from Modules.cnn_vae import CNN_VAE
from Modules.cnn_vae_couple import CNN_VAE_COUPLE
from Modules.Losses.SeqLoss import SeqLoss
from Modules.Losses.Reward import RewardCriterion
from Modules.Losses.GANLoss import GANLoss
from tqdm import tqdm
from dataloaders.ptb import PTB, PTB_Interp
from dataloaders.yelp import Yelp, Yelp_Interp, YelpWithLabel
from dataloaders.yahoo import Yahoo, Yahoo_Interp
from utils.evaluations import Evaluator
from config import Config
from utils.utils import gpu_wrapper, strip_pad, pretty_string, sample_2d, strip_eos, create_null_mask
from utils.multi_bleu import calc_bleu_score
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
plt.switch_backend('agg')


import seaborn as sns
sns.set(color_codes=True)
sns.set(style="white")

config = Config()
smooth = SmoothingFunction()
ROUND = config.ROUND
EPSILON = 1e-10
bi_TS = 0.2
tri_TS = 0.1
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)


class Experiment(object):

    def __init__(self, test=False):
        self.is_test = test

        print('----- Loading data -----')
        if config.dataset == 'ptb':
            if not self.is_test:
                self.train_set = PTB('train')
            self.test_set = PTB('test')
            self.val_set = PTB('valid')
        elif config.dataset == 'yelp':
            if not self.is_test:
                self.train_set = Yelp('train')
            self.test_set = Yelp('test')
            self.val_set = Yelp('valid')
        elif config.dataset == 'yahoo':
            if not self.is_test:
                self.train_set = Yahoo('train')
            self.test_set = Yahoo('test')
            self.val_set = Yahoo('valid')
        else:
            raise ValueError()
        if not self.is_test:
            print('The train set has {} items'.format(len(self.train_set)))
        print('The test set has {} items'.format(len(self.test_set)))
        print('The val set has {} items'.format(len(self.val_set)))
        # for i in range(20):
        #     self.train_set[i]

        self.vocab = self.test_set.vocab
        self.evaluator = Evaluator(dataset=config.dataset, vocab=self.vocab)

        print('----- Loading model -----')
        Enc_Emb = nn.Embedding.from_pretrained(self.vocab.embedding.clone(), freeze=False)
        Dec_Emb = Enc_Emb

        if config.model == 'lm':
            self.AutoEncoder = LM(hid_dim=config.hid_dim,
                                  latent_dim=config.latent_dim,  # Kept for nothing.
                                  dec_layers=config.dec_layers,
                                  dropout=config.dropout,
                                  dec_max_len=config.max_len,
                                  beam_size=config.beam_size,
                                  WEAtt_type=config.WEAtt_type,
                                  decoder_emb=Dec_Emb,
                                  pad_id=self.test_set.pad)
        elif config.model == 'dae':
            self.AutoEncoder = DAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)
        elif config.model == 'vae':
            self.AutoEncoder = VAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 'vae-couple':
            self.AutoEncoder = VAE_COUPLE(hid_dim=config.hid_dim,
                                          latent_dim=config.latent_dim,
                                          enc_layers=config.enc_layers,
                                          dec_layers=config.dec_layers,
                                          dropout=config.dropout,
                                          enc_bi=True,
                                          dec_max_len=config.max_len,
                                          beam_size=config.beam_size,
                                          WEAtt_type=config.WEAtt_type,
                                          encoder_emb=Enc_Emb,
                                          decoder_emb=Dec_Emb,
                                          pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 'beta-vae':
            self.AutoEncoder = VAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        elif config.model == 'beta-vae-couple':
            self.AutoEncoder = VAE_COUPLE(hid_dim=config.hid_dim,
                                          latent_dim=config.latent_dim,
                                          enc_layers=config.enc_layers,
                                          dec_layers=config.dec_layers,
                                          dropout=config.dropout,
                                          enc_bi=True,
                                          dec_max_len=config.max_len,
                                          beam_size=config.beam_size,
                                          WEAtt_type=config.WEAtt_type,
                                          encoder_emb=Enc_Emb,
                                          decoder_emb=Dec_Emb,
                                          pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        elif config.model == 'vae-nf':
            self.AutoEncoder = VAE_NF(hid_dim=config.hid_dim,
                                      latent_dim=config.latent_dim,
                                      enc_layers=config.enc_layers,
                                      dec_layers=config.dec_layers,
                                      dropout=config.dropout,
                                      enc_bi=True,
                                      dec_max_len=config.max_len,
                                      beam_size=config.beam_size,
                                      WEAtt_type=config.WEAtt_type,
                                      encoder_emb=Enc_Emb,
                                      decoder_emb=Dec_Emb,
                                      pad_id=self.test_set.pad,
                                      n_flows=config.n_flows,
                                      flow_type=config.flow_type)
            self.lambda_KL = 0
        elif config.model == 'vae-nf-couple':
            self.AutoEncoder = VAE_NF_COUPLE(hid_dim=config.hid_dim,
                                             latent_dim=config.latent_dim,
                                             enc_layers=config.enc_layers,
                                             dec_layers=config.dec_layers,
                                             dropout=config.dropout,
                                             enc_bi=True,
                                             dec_max_len=config.max_len,
                                             beam_size=config.beam_size,
                                             WEAtt_type=config.WEAtt_type,
                                             encoder_emb=Enc_Emb,
                                             decoder_emb=Dec_Emb,
                                             pad_id=self.test_set.pad,
                                             n_flows=config.n_flows,
                                             flow_type=config.flow_type)
            self.lambda_KL = 0
        elif config.model == 'wae':
            self.AutoEncoder = WAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 'wae-couple':
            self.AutoEncoder = WAE_COUPLE(hid_dim=config.hid_dim,
                                          latent_dim=config.latent_dim,
                                          enc_layers=config.enc_layers,
                                          dec_layers=config.dec_layers,
                                          dropout=config.dropout,
                                          enc_bi=True,
                                          dec_max_len=config.max_len,
                                          beam_size=config.beam_size,
                                          WEAtt_type=config.WEAtt_type,
                                          encoder_emb=Enc_Emb,
                                          decoder_emb=Dec_Emb,
                                          pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 'wae-nf':
            self.AutoEncoder = WAE_NF(hid_dim=config.hid_dim,
                                      latent_dim=config.latent_dim,
                                      enc_layers=config.enc_layers,
                                      dec_layers=config.dec_layers,
                                      dropout=config.dropout,
                                      enc_bi=True,
                                      dec_max_len=config.max_len,
                                      beam_size=config.beam_size,
                                      WEAtt_type=config.WEAtt_type,
                                      encoder_emb=Enc_Emb,
                                      decoder_emb=Dec_Emb,
                                      pad_id=self.test_set.pad,
                                      n_flows=config.n_flows,
                                      flow_type=config.flow_type)
            self.lambda_KL = 0
        elif config.model == 'wae-nf-couple':
            self.AutoEncoder = WAE_NF_COUPLE(hid_dim=config.hid_dim,
                                             latent_dim=config.latent_dim,
                                             enc_layers=config.enc_layers,
                                             dec_layers=config.dec_layers,
                                             dropout=config.dropout,
                                             enc_bi=True,
                                             dec_max_len=config.max_len,
                                             beam_size=config.beam_size,
                                             WEAtt_type=config.WEAtt_type,
                                             encoder_emb=Enc_Emb,
                                             decoder_emb=Dec_Emb,
                                             pad_id=self.test_set.pad,
                                             n_flows=config.n_flows,
                                             flow_type=config.flow_type)
            self.lambda_KL = 0
        elif config.model == 's-vae':
            self.AutoEncoder = S_VAE(hid_dim=config.hid_dim,
                                     latent_dim=config.latent_dim,
                                     enc_layers=config.enc_layers,
                                     dec_layers=config.dec_layers,
                                     dropout=config.dropout,
                                     enc_bi=True,
                                     dec_max_len=config.max_len,
                                     beam_size=config.beam_size,
                                     WEAtt_type=config.WEAtt_type,
                                     encoder_emb=Enc_Emb,
                                     decoder_emb=Dec_Emb,
                                     pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 's-vae-couple':
            self.AutoEncoder = S_VAE_COUPLE(hid_dim=config.hid_dim,
                                            latent_dim=config.latent_dim,
                                            enc_layers=config.enc_layers,
                                            dec_layers=config.dec_layers,
                                            dropout=config.dropout,
                                            enc_bi=True,
                                            dec_max_len=config.max_len,
                                            beam_size=config.beam_size,
                                            WEAtt_type=config.WEAtt_type,
                                            encoder_emb=Enc_Emb,
                                            decoder_emb=Dec_Emb,
                                            pad_id=self.test_set.pad)
            self.lambda_KL = 0
        elif config.model == 'cnn-vae':
            self.AutoEncoder = CNN_VAE(hid_dim=config.hid_dim,
                                       latent_dim=config.latent_dim,
                                       enc_layers=config.enc_layers,
                                       dec_layers=config.dec_layers,
                                       dropout=config.dropout,
                                       enc_bi=True,
                                       dec_max_len=config.max_len,
                                       beam_size=config.beam_size,
                                       WEAtt_type=config.WEAtt_type,
                                       encoder_emb=Enc_Emb,
                                       decoder_emb=Dec_Emb,
                                       pad_id=self.test_set.pad,
                                       scale=config.scale)
            self.lambda_KL = 0
        elif config.model == 'cnn-vae-couple':
            self.AutoEncoder = CNN_VAE_COUPLE(hid_dim=config.hid_dim,
                                              latent_dim=config.latent_dim,
                                              enc_layers=config.enc_layers,
                                              dec_layers=config.dec_layers,
                                              dropout=config.dropout,
                                              enc_bi=True,
                                              dec_max_len=config.max_len,
                                              beam_size=config.beam_size,
                                              WEAtt_type=config.WEAtt_type,
                                              encoder_emb=Enc_Emb,
                                              decoder_emb=Dec_Emb,
                                              pad_id=self.test_set.pad,
                                              scale=config.scale)
            self.lambda_KL = 0
        elif config.model == 'cyc-anneal-vae':
            self.AutoEncoder = VAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        elif config.model == 'cyc-anneal-vae-couple':
            self.AutoEncoder = VAE_COUPLE(hid_dim=config.hid_dim,
                                          latent_dim=config.latent_dim,
                                          enc_layers=config.enc_layers,
                                          dec_layers=config.dec_layers,
                                          dropout=config.dropout,
                                          enc_bi=True,
                                          dec_max_len=config.max_len,
                                          beam_size=config.beam_size,
                                          WEAtt_type=config.WEAtt_type,
                                          encoder_emb=Enc_Emb,
                                          decoder_emb=Dec_Emb,
                                          pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        elif config.model == 'surprising-fix':
            self.AutoEncoder = VAE(hid_dim=config.hid_dim,
                                   latent_dim=config.latent_dim,
                                   enc_layers=config.enc_layers,
                                   dec_layers=config.dec_layers,
                                   dropout=config.dropout,
                                   enc_bi=True,
                                   dec_max_len=config.max_len,
                                   beam_size=config.beam_size,
                                   WEAtt_type=config.WEAtt_type,
                                   encoder_emb=Enc_Emb,
                                   decoder_emb=Dec_Emb,
                                   pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        elif config.model == 'surprising-fix-couple':
            self.AutoEncoder = VAE_COUPLE(hid_dim=config.hid_dim,
                                          latent_dim=config.latent_dim,
                                          enc_layers=config.enc_layers,
                                          dec_layers=config.dec_layers,
                                          dropout=config.dropout,
                                          enc_bi=True,
                                          dec_max_len=config.max_len,
                                          beam_size=config.beam_size,
                                          WEAtt_type=config.WEAtt_type,
                                          encoder_emb=Enc_Emb,
                                          decoder_emb=Dec_Emb,
                                          pad_id=self.test_set.pad)  # TODO: not a bug here. The same architecture is used.
            self.lambda_KL = 0
        else:
            raise ValueError()

        # Restore pretrained posterior network.
        if config.model in ['surprising-fix', 'surprising-fix-couple']:
            load_path = os.path.join('outputs', 'saved_model', '{}-dae'.format(config.dataset), 'best-AutoEncoder.ckpt')
            pretrained_dae = DAE(hid_dim=config.hid_dim,
                                 latent_dim=config.latent_dim,
                                 enc_layers=config.enc_layers,
                                 dec_layers=config.dec_layers,
                                 dropout=config.dropout,
                                 enc_bi=True,
                                 dec_max_len=config.max_len,
                                 beam_size=config.beam_size,
                                 WEAtt_type=config.WEAtt_type,
                                 encoder_emb=Enc_Emb,
                                 decoder_emb=Dec_Emb,
                                 pad_id=self.test_set.pad)
            pretrained_dae.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage), strict=True)
            self.AutoEncoder.Encoder.Embedding.weight.data.copy_(pretrained_dae.Encoder.Embedding.weight.data)
            self.AutoEncoder.Encoder.GRU.weight_ih_l0.data.copy_(pretrained_dae.Encoder.GRU.weight_ih_l0.data)
            self.AutoEncoder.Encoder.GRU.weight_hh_l0.data.copy_(pretrained_dae.Encoder.GRU.weight_hh_l0.data)
            self.AutoEncoder.Encoder.GRU.bias_ih_l0.data.copy_(pretrained_dae.Encoder.GRU.bias_ih_l0.data)
            self.AutoEncoder.Encoder.GRU.bias_hh_l0.data.copy_(pretrained_dae.Encoder.GRU.bias_hh_l0.data)
            self.AutoEncoder.Encoder.GRU.weight_ih_l0_reverse.data.copy_(pretrained_dae.Encoder.GRU.weight_ih_l0_reverse.data)
            self.AutoEncoder.Encoder.GRU.weight_hh_l0_reverse.data.copy_(pretrained_dae.Encoder.GRU.weight_hh_l0_reverse.data)
            self.AutoEncoder.Encoder.GRU.bias_ih_l0_reverse.data.copy_(pretrained_dae.Encoder.GRU.bias_ih_l0_reverse.data)
            self.AutoEncoder.Encoder.GRU.bias_hh_l0_reverse.data.copy_(pretrained_dae.Encoder.GRU.bias_hh_l0_reverse.data)
            getattr(self.AutoEncoder.PosteriorGaussian.toMiu, '0').weight.data.copy_(pretrained_dae.toLatent.weight.data)
            getattr(self.AutoEncoder.PosteriorGaussian.toMiu, '0').bias.data.copy_(pretrained_dae.toLatent.bias.data)

        if config.model in ['lm', 'dae',
                            'vae', 'vae-couple',
                            'beta-vae', 'beta-vae-couple',
                            'vae-nf', 'vae-nf-couple',
                            'wae', 'wae-couple',
                            'wae-nf', 'wae-nf-couple',
                            's-vae', 's-vae-couple',
                            'cnn-vae', 'cnn-vae-couple',
                            'cyc-anneal-vae', 'cyc-anneal-vae-couple',
                            'surprising-fix', 'surprising-fix-couple',
                            ]:
            self.modules = ['AutoEncoder']  # TODO
        elif config.model in []:
            self.modules = ['AutoEncoder', 'Discriminator']  # TODO
        else:
            raise ValueError()
        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))

        if config.train_mode == 'gen':
            if config.model in ['lm', 'dae',
                                'vae', 'vae-couple',
                                'beta-vae', 'beta-vae-couple',
                                'vae-nf', 'vae-nf-couple',
                                'wae', 'wae-couple',
                                'wae-nf', 'wae-nf-couple',
                                's-vae', 's-vae-couple',
                                'cnn-vae', 'cnn-vae-couple',
                                'cyc-anneal-vae', 'cyc-anneal-vae-couple',
                                'surprising-fix', 'surprising-fix-couple',
                                ]:
                self.scopes = {'gen': ['AutoEncoder']}  # TODO
            elif config.model in []:
                self.scopes = {
                    'gen': ['AutoEncoder'],
                    'dis': ['Discriminator']
                }  # TODO
            else:
                raise ValueError()
        else:
            raise ValueError()
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = 0
        self.logger = None
        if config.train_mode == 'gen':
            self.best_metric = float('inf')
        else:
            raise ValueError()
        self.criterionSeq, self.criterionReward, self.criterionCls, self.criterionBernKL, self.criterionGAN = None, None, None, None, None
        self.cached_adv_gen = 0

    def restore_from(self, module, path):
        print('Loading the trained best models...')
        path = os.path.join(path, 'best-{}.ckpt'.format(module))
        getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=True)

    def restore_model(self, modules):
        print('Loading the trained best models...')
        for module in modules:
            path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
            getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                  strict=True)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from utils.logger import Logger
        self.logger = Logger(config.log_dir)

    def log_step(self, loss):
        # Log loss.
        for loss_name, value in loss.items():
            self.logger.scalar_summary(loss_name, value, self.iter_num)
        # Log learning rate.
        for scope in self.scopes:
            self.logger.scalar_summary('{}/lr'.format(scope), getattr(self, scope + '_lr'), self.iter_num)

    def save_step(self, modules, use_iter=False):
        save_dir = config.save_model_dir
        if use_iter:
            for module in modules:
                path = os.path.join(save_dir, '{}-{}.ckpt'.format(self.iter_num, module))
                torch.save(getattr(self, module).state_dict(), path)
        else:
            for module in modules:
                path = os.path.join(save_dir, 'best-{}.ckpt'.format(module))
                torch.save(getattr(self, module).state_dict(), path)
        print('Saved model checkpoints into {}...\n\n\n\n\n\n\n\n\n\n\n\n'.format(save_dir))

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes):
        trainable = []
        for scope in scopes:
            trainable.extend(getattr(self, 'trainable_' + scope))
        # Clip on all parameters.
        if config.clip_norm < float('inf'):
            clip_grad_norm_(parameters=trainable, max_norm=config.clip_norm)
        if config.clip_value < float('inf'):
            clip_value = float(config.clip_value)
            for p in filter(lambda p: p.grad is not None, trainable):
                p.grad.data.clamp_(min=-clip_value, max=clip_value)
        # Backward.
        for scope in scopes:
            getattr(self, scope + '_optim').step()

    def update_lambda_kl(self):
        if config.kl_annealing:
            if config.model in ['beta-vae', 'beta-vae-couple']:
                max_val = config.beta
            elif config.model in ['lm', 'dae',
                                  'vae', 'vae-couple',
                                  'vae-nf', 'vae-nf-couple',
                                  's-vae', 's-vae-couple',
                                  'cnn-vae', 'cnn-vae-couple',
                                  'surprising-fix', 'surprising-fix-couple',
                                  ]:
                max_val = 1
            elif config.model in ['wae', 'wae-couple',
                                  'wae-nf', 'wae-nf-couple',
                                  ]:
                max_val = 0.8
            elif config.model in ['cyc-anneal-vae', 'cyc-anneal-vae-couple',
                                  ]:
                if self.iter_num < config.annealing_start:
                    self.lambda_KL = 0
                elif self.iter_num > config.annealing_step + config.annealing_start:
                    self.lambda_KL = 1
                else:
                    tau = ((self.iter_num - config.annealing_start) % int(config.annealing_step / config.M)) / (config.annealing_step / config.M)
                    if tau > config.R:
                        self.lambda_KL = 1
                    else:
                        self.lambda_KL = tau / config.R
                # FIXME: the code is correct, but a bit confusing.
                return
            else:
                raise ValueError()
            self.lambda_KL = min(max(np.round((self.iter_num - config.annealing_start) / config.annealing_step, decimals=6), 0), max_val)
        else:
            self.lambda_KL = 1

    def update_lr(self):
        for scope in self.scopes:
            setattr(self, scope + '_lr', getattr(self, scope + '_lr') / 2)  # Half the learning rate.
            for param_group in getattr(self, scope + '_optim').param_groups:
                param_group['lr'] = getattr(self, scope + '_lr')

    def set_requires_grad(self, modules, requires_grad):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = requires_grad

    def set_training(self, mode):
        for module in self.modules:
            if getattr(self, module) is not None:
                getattr(self, module).train(mode=mode)

    def train(self):

        # Logging.
        if config.use_tensorboard:
            self.build_tensorboard()

        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            trainable = []
            for module in self.scopes[scope]:
                if getattr(self, module) is not None:
                    for k, v in getattr(self, module).state_dict(keep_vars=True).items():
                        # k is the parameter name; v is the parameter value.
                        if v.requires_grad:
                            trainable.append(v)
                            print("[{} Trainable:]".format(module), k)
                        else:
                            print("[{} Frozen:]".format(module), k)
            setattr(self, scope + '_optim', Adam(params=trainable,
                                                 lr=getattr(self, scope + '_lr'),
                                                 betas=[config.beta1, config.beta2],
                                                 weight_decay=config.weight_decay))
            setattr(self, 'trainable_' + scope, trainable)

        # Build criterion.
        self.criterionSeq = SeqLoss(voc_size=self.train_set.vocab.size, pad=self.train_set.pad,
                                    end=self.train_set.eos, unk=self.train_set.unk)
        # self.criterionReward = RewardCriterion()
        # self.criterionGAN = GANLoss(config.gan_type)
        # self.criterionCls = F.binary_cross_entropy_with_logits
        # self.criterionBernKL = F.binary_cross_entropy

        # Train.
        epoch = 0
        # self.language_modeling('test')
        try:
            while True:
                self.train_epoch(epoch_idx=epoch)
                epoch += 1
                if self.iter_num >= config.num_iters:
                    break
        except KeyboardInterrupt:
            print('-' * 100)
            print('Quit training.')

    def test(self):
        if config.train_mode == 'gen':
            self.restore_model(['AutoEncoder'])
        else:
            raise ValueError()

        # Language modeling test.
        self.language_modeling(val_or_test='test', save=True)
        # Sample from prior test.
        # for beam_size in config.enum_beam_size:
        #     self.sample_from_prior(beam_size=beam_size, save=True)
        self.sample_from_prior(beam_size=1, save=True)

    def val(self):
        raise NotImplementedError()

    def train_epoch(self, epoch_idx):

        loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.set_training(mode=True)

        with tqdm(loader) as pbar:
            for data in pbar:
                self.iter_num += 1
                loss = {}

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                data = self.preprocess_data(data)

                # =================================================================================== #
                #                                       2. Training                                   #
                # =================================================================================== #

                if config.train_mode == 'gen':
                    bare, go, eos, sent_len = data

                    max_len = max(sent_len).item()
                    bare = bare[:, :max_len].contiguous()
                    go = go[:, :max_len + 1].contiguous()
                    eos = eos[:, :max_len + 1].contiguous()

                    if config.model == 'lm':
                        logits = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        tot_loss = seq_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'dae':
                        logits, latent_vectors = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        tot_loss = seq_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model in ['vae', 'beta-vae', 'cyc-anneal-vae']:
                        logits, pred_gaussian, latent_vectors, BoW_logits = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian, standard_gaussian)  # shape = (n_batch, latent_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1],
                                                                        -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + self.lambda_KL * kl_loss

                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model in ['vae-couple', 'beta-vae-couple', 'cyc-anneal-vae-couple']:
                        logits, pred_gaussian, latent_vectors, BoW_logits, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian, standard_gaussian)  # shape = (n_batch, emb_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1],
                                                                        -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        if config.model == 'vae-couple' and config.euclidean_dist:
                            couple_loss = euc_init
                        else:
                            couple_loss = imq_kernel_dist

                        # Batch mean.
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                                   self.lambda_KL * kl_loss + \
                                   seq_loss_couple * config.lambda_seq_couple + \
                                   config.lambda_couple * couple_loss

                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'vae-nf':
                        logits, Q0, z0, zk, sum_log_jacobian, BoW_logits = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1], -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        # Prior distribution.
                        Pk = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))

                        # Batch mean.
                        Q0z0 = Q0.log_prob(z0).sum(1).mean(0)
                        Pkzk = Pk.log_prob(zk).sum(1).mean(0)
                        sudo_kl = Q0z0 - Pkzk
                        sum_log_jacobian = sum_log_jacobian.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                            self.lambda_KL * (sudo_kl - sum_log_jacobian)
                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['Q0z0/L'] = round(Q0z0.item(), ROUND)
                        loss['Pkzk/L'] = round(Pkzk.item(), ROUND)
                        # loss['sudo_kl/L'] = round(sudo_kl.item(), ROUND)
                        loss['jacob/L'] = round(sum_log_jacobian.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'vae-nf-couple':
                        logits, Q0, z0, zk, sum_log_jacobian, BoW_logits, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1], -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        couple_loss = imq_kernel_dist

                        # Prior distribution.
                        Pk = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                        gpu_wrapper(torch.ones(config.latent_dim)))

                        # Batch mean.
                        Q0z0 = Q0.log_prob(z0).sum(1).mean(0)
                        Pkzk = Pk.log_prob(zk).sum(1).mean(0)
                        sudo_kl = Q0z0 - Pkzk
                        sum_log_jacobian = sum_log_jacobian.mean(0)
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                                   self.lambda_KL * (sudo_kl - sum_log_jacobian) + \
                                   seq_loss_couple + \
                                   config.lambda_couple * couple_loss

                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['Q0z0/L'] = round(Q0z0.item(), ROUND)
                        loss['Pkzk/L'] = round(Pkzk.item(), ROUND)
                        # loss['sudo_kl/L'] = round(sudo_kl.item(), ROUND)
                        loss['jacob/L'] = round(sum_log_jacobian.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'wae':

                        logits, gaussian_dist, latent_vectors = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        mmd_loss = compute_mmd(latent_vectors)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(gaussian_dist, standard_gaussian)  # shape = (n_batch, emb_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + config.lambda_mmd * mmd_loss + self.lambda_KL * kl_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['mmd/L'] = round(mmd_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'wae-couple':
                        logits, gaussian_dist, latent_vectors, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)
                        mmd_loss = compute_mmd(latent_vectors)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(gaussian_dist, standard_gaussian)  # shape = (n_batch, emb_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        couple_loss = imq_kernel_dist

                        # Batch mean.
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                                   config.lambda_mmd * mmd_loss + \
                                   self.lambda_KL * kl_loss + \
                                   seq_loss_couple + \
                                   config.lambda_couple * couple_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['mmd/L'] = round(mmd_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'wae-nf':
                        logits, Q0, z0, zk, sum_log_jacobian = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        mmd_loss = compute_mmd(zk)

                        # Prior distribution.
                        Pk = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)), gpu_wrapper(torch.ones(config.latent_dim)))

                        # Batch mean.
                        Q0z0 = Q0.log_prob(z0).sum(1).mean(0)
                        Pkzk = Pk.log_prob(zk).sum(1).mean(0)
                        sudo_kl = Q0z0 - Pkzk
                        sum_log_jacobian = sum_log_jacobian.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                            config.lambda_mmd * mmd_loss + \
                            self.lambda_KL * (sudo_kl - sum_log_jacobian)

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['mmd/L'] = round(mmd_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['Q0z0/L'] = round(Q0z0.item(), ROUND)
                        loss['Pkzk/L'] = round(Pkzk.item(), ROUND)
                        # loss['sudo_kl/L'] = round(sudo_kl.item(), ROUND)
                        loss['jacob/L'] = round(sum_log_jacobian.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'wae-nf-couple':
                        logits, Q0, z0, zk, sum_log_jacobian, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)
                        mmd_loss = compute_mmd(zk)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        couple_loss = imq_kernel_dist

                        # Prior distribution.
                        Pk = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                        gpu_wrapper(torch.ones(config.latent_dim)))

                        # Batch mean.
                        Q0z0 = Q0.log_prob(z0).sum(1).mean(0)
                        Pkzk = Pk.log_prob(zk).sum(1).mean(0)
                        sudo_kl = Q0z0 - Pkzk
                        sum_log_jacobian = sum_log_jacobian.mean(0)
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                            config.lambda_mmd * mmd_loss + \
                            self.lambda_KL * (sudo_kl - sum_log_jacobian) + \
                            seq_loss_couple + \
                            config.lambda_couple * couple_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['mmd/L'] = round(mmd_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['Q0z0/L'] = round(Q0z0.item(), ROUND)
                        loss['Pkzk/L'] = round(Pkzk.item(), ROUND)
                        # loss['sudo_kl/L'] = round(sudo_kl.item(), ROUND)
                        loss['jacob/L'] = round(sum_log_jacobian.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 's-vae':
                        logits, posterior_vmf, prior_unif = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)

                        kl_loss = torch.distributions.kl_divergence(posterior_vmf, prior_unif)  # shape = (n_batch, )
                        kl_loss = kl_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + self.lambda_KL * kl_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 's-vae-couple':
                        logits, posterior_vmf, prior_unif, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)

                        kl_loss = torch.distributions.kl_divergence(posterior_vmf, prior_unif)  # shape = (n_batch, )
                        kl_loss = kl_loss.mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        couple_loss = imq_kernel_dist

                        # Batch mean.
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                            self.lambda_KL * kl_loss + \
                            seq_loss_couple + \
                            config.lambda_couple * couple_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'cnn-vae':
                        logits, pred_gaussian, latent_vectors = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian, standard_gaussian)  # shape = (n_batch, latent_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + self.lambda_KL * kl_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'cnn-vae-couple':
                        logits, pred_gaussian, latent_vectors, logits_couple, init, init_couple = self.AutoEncoder(bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian,
                                                                    standard_gaussian)  # shape = (n_batch, emb_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        couple_loss = imq_kernel_dist

                        # Batch mean.
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                            self.lambda_KL * kl_loss + \
                            seq_loss_couple + \
                            config.lambda_couple * couple_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model == 'surprising-fix':
                        logits, pred_gaussian, latent_vectors, BoW_logits = self.AutoEncoder(bare=bare,
                                                                                             sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian,
                                                                    standard_gaussian)  # shape = (n_batch, latent_dim)
                        # FB.
                        kl_loss = torch.clamp(kl_loss, min=config.lambda_fb / config.latent_dim).sum(1).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1],
                                                                        -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + self.lambda_KL * kl_loss

                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['KL/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])
                    elif config.model == 'surprising-fix-couple':
                        logits, pred_gaussian, latent_vectors, BoW_logits, logits_couple, init, init_couple = self.AutoEncoder(
                            bare=bare, sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).mean(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian,
                                                                    standard_gaussian)  # shape = (n_batch, emb_dim)
                        # FB.
                        kl_loss = torch.clamp(kl_loss, min=config.lambda_fb / config.latent_dim).sum(1).mean(0)

                        if config.lambda_BoW > 0:
                            BoW_logits = BoW_logits.unsqueeze(1).expand(-1, bare.shape[1],
                                                                        -1)  # shape = (n_batch, 15, V)
                            bow_loss = (self.criterionSeq(BoW_logits, bare, keep_batch=True) / sent_len.float()).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        if config.model == 'vae-couple' and config.euclidean_dist:
                            couple_loss = euc_init
                        else:
                            couple_loss = imq_kernel_dist

                        # Batch mean.
                        couple_loss = couple_loss.mean(0)

                        self.update_lambda_kl()
                        tot_loss = seq_loss + \
                                   self.lambda_KL * kl_loss + \
                                   seq_loss_couple * config.lambda_seq_couple + \
                                   config.lambda_couple * couple_loss

                        if config.lambda_BoW > 0:
                            if self.lambda_KL > 1e-5:
                                tot_loss = tot_loss + config.lambda_BoW * bow_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)
                        loss['seq_c/L'] = round(seq_loss_couple.item(), ROUND)
                        if config.lambda_BoW > 0:
                            loss['bow/L'] = round(bow_loss.item(), ROUND)
                        loss['couple/L'] = round(couple_loss.item(), ROUND)
                        loss['kl/lambda'] = round(self.lambda_KL, ROUND)
                        loss['KL/L'] = round(kl_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])
                    else:
                        raise ValueError()

                else:
                    raise ValueError()

                # =================================================================================== #
                #                                 6. Miscellaneous                                    #
                # =================================================================================== #

                if self.iter_num % 1 != 0:
                    for k in loss.keys():
                        print(k + ':' + pretty_string(loss[k]))
                    print()

                display = ', '.join([key + ':' + pretty_string(loss[key]) for key in loss.keys()])
                pbar.set_description_str(display)

                # Print out training information.
                if self.iter_num % config.log_step == 0 and config.use_tensorboard:
                    self.log_step(loss)

                # Validation.
                if self.iter_num % config.sample_step == 0:
                    # self.sample_from_prior()
                    self.language_modeling('val')

                # Decay learning rates.
                if self.iter_num % config.lr_decay_step == 0 and self.iter_num > config.start_decay:
                    self.update_lr()

    def sample_from_prior(self, beam_size=config.beam_size, save=False):
        if config.model == 'lm':
            print('LM does not support sampling.')
            return

        self.AutoEncoder.Decoder.beam_size = beam_size
        n_sample_batch = 100
        loader = range(n_sample_batch)

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'preds': [],
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    preds = self.AutoEncoder.sample_from_prior(go=gpu_wrapper(torch.zeros([config.batch_size, 1])).long() + self.test_set.go)
                    batchify['preds'].append(preds)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp
            batchify['preds'] = [strip_eos([self.vocab.id2word[idx.item()] for idx in pred]) for pred in batchify['preds']]

            # To save results.
            to_write = []

            # Diversity metrics.
            dist_1, dist_2 = self.evaluator.eval_DIST(batchify['preds'])
            to_write.append('Dist 1 = {}'.format(round(dist_1, ROUND)))
            print('\n\n\nDist 1 = {}'.format(round(dist_1, ROUND)))
            to_write.append('Dist 2 = {}'.format(round(dist_2, ROUND)))
            print('Dist 2 = {}'.format(round(dist_2, ROUND)))

            use_metric = [False, True][0]
            if dist_2 > self.best_metric or not use_metric:

                # ----- Peep the resutls -----
                peep = 10
                for pred in batchify['preds'][:peep]:
                    print('PRED        = ' + ' '.join(pred))
                    print('-' * 50)
                for idx in range(3):
                    print('*' * 150)

                # ----- Save the results to file -----
                save_num = 40
                if save:
                    save_file = os.path.join(config.sample_dir, 'sample_from_prior.txt')
                    if os.path.exists(save_file):
                        print('WARNING. The file sample_from_prior.txt exists.')
                    with open(save_file, 'w') as f:
                        for item in to_write:
                            f.write(item + '\n')
                        f.write('*' * 150 + '\n')
                        for pred in batchify['preds'][:save_num]:
                            f.write('PRED        = ' + ' '.join(pred) + '\n')
                            f.write('-' * 50 + '\n')

        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    def language_modeling(self, val_or_test, save=False):
        dataset = {
            "test": self.test_set,
            "val": self.val_set
        }[val_or_test]
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)
        if val_or_test == 'val':
            n_sample = 10
        elif val_or_test == 'test':
            n_sample = 100
        else:
            raise ValueError()

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'xent': [],
                'nll': [],
                'kl': [],
                'latents': [],
                'sent_len': []
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    data = self.preprocess_data(data)
                    bare, go, eos, sent_len = data

                    xent, nll, kl, latent_vectors = self.AutoEncoder.test_lm(bare=bare,
                                                                             sent_len=sent_len,
                                                                             go=go,
                                                                             eos=eos,
                                                                             n_sample=n_sample)

                    batchify['xent'].append(xent)
                    batchify['nll'].append(nll)
                    batchify['kl'].append(kl)
                    batchify['latents'].append(latent_vectors)
                    batchify['sent_len'].append(sent_len)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp
            # batchify['preds'] = [strip_eos([self.vocab.id2word[idx.item()] for idx in pred]) for pred in batchify['preds']]

            # ---- Get original articles and references -----
            sents = dataset.get_sentences()

            # To save results.
            to_write = []

            # Language modeling metrics.
            avg_nll = batchify['nll'].mean(0).item()
            to_write.append('nll = {}'.format(avg_nll))
            print('\n\n\nnll = {}'.format(avg_nll))

            avg_xent = batchify['xent'].mean(0).item()
            to_write.append('xent = {}'.format(avg_xent))
            print('\n\n\nxent = {}'.format(avg_xent))

            avg_kl = batchify['kl'].mean(0).item()
            to_write.append('kl = {}'.format(avg_kl))
            print('kl = {}'.format(avg_kl))

            ppl = torch.exp(batchify['nll'].sum(0) / (batchify['sent_len'] + 1).float().sum(0)).item()
            to_write.append('ppl = {}'.format(ppl))
            print('ppl = {}'.format(ppl))

            use_metric = [False, True][1]
            if config.model == 'dae':
                metric = avg_xent
            else:
                metric = avg_nll
            if metric < self.best_metric or not use_metric:
                if not use_metric:
                    print('Save anyway.')
                    self.save_step(['AutoEncoder'])
                else:
                    self.best_metric = metric
                    print('New best metric found.')
                    self.save_step(['AutoEncoder'])

                # ----- Peep the resutls -----
                peep = 10
                for latent, xent, nll, kl, sent in zip(batchify['latents'][:peep],
                                                       batchify['xent'][:peep],
                                                       batchify['nll'][:peep],
                                                       batchify['kl'][:peep],
                                                       sents[:peep]):
                    print('SENT              = ' + ' '.join(sent))
                    print('NLL               = {}'.format(nll.item()))
                    print('XENT              = {}'.format(xent.item()))
                    print('KL                = {}'.format(kl.item()))
                    print('-' * 50)
                for idx in range(3):
                    print('*' * 150)

                # ----- Save the results to file -----
                save_num = 40
                if save:
                    save_file = os.path.join(config.sample_dir, 'language_modeling.txt')
                    if os.path.exists(save_file):
                        print('WARNING. The file language_modeling.txt exists.')
                    with open(save_file, 'w') as f:
                        for item in to_write:
                            f.write(item + '\n')
                        f.write('*' * 150 + '\n')
                        for latent, xent, nll, kl, sent in zip(batchify['latents'][:save_num],
                                                               batchify['xent'][:save_num],
                                                               batchify['nll'][:save_num],
                                                               batchify['kl'][:save_num],
                                                               sents[:peep]):
                            f.write('SENT              = ' + ' '.join(sent) + '\n')
                            f.write('XENT              = {}'.format(xent.item()) + '\n')
                            f.write('NLL               = {}'.format(nll.item()) + '\n')
                            f.write('KL                = {}'.format(kl.item()) + '\n')
                            f.write('-' * 50 + '\n')
        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    def estimate_mi(self):
        if config.model in ['vae', 'vae-couple',
                            'vae-nf', 'vae-nf-couple',
                            'wae', 'wae-couple',
                            'wae-nf', 'wae-nf-couple',
                            'beta-vae', 'beta-vae-couple',
                            's-vae', 's-vae-couple',
                            'cnn-vae', 'cnn-vae-couple',
                            'cyc-anneal-vae', 'cyc-anneal-vae-couple',
                            'surprising-fix', 'surprising-fix-couple',
                            ]:
            pass
        elif config.model == 'lm':
            print('LM does not support MI estimation.')
        else:
            raise ValueError()

        dataset = self.test_set
        loader = DataLoader(dataset, batch_size=512 + 1, shuffle=False, num_workers=config.num_workers)
        n_sample = 100

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'mi': []
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    data = self.preprocess_data(data)
                    bare, go, eos, sent_len = data

                    mi, sampled_latents = self.AutoEncoder.estimate_mi(bare=bare,
                                                                       sent_len=sent_len,
                                                                       go=go,
                                                                       n_sample=n_sample)

                    batchify['mi'].append(mi)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp

            avg_mi = batchify['mi'].mean(0).item()
            print('\n\n\nMutual Information = {}'.format(avg_mi))
            print(config.save_model_dir)

        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    def analyze_saliency(self, val_or_test):
        if config.model in ['vae', 'vae-couple', 'beta-vae', 'beta-vae-couple', 'dae']:
            pass
        else:
            raise ValueError()

        dataset = {
            "test": self.test_set,
            "val": self.val_set
        }[val_or_test]
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=config.num_workers)
        n_sample = 100

        self.criterionSeq = SeqLoss(voc_size=self.test_set.vocab.size, pad=self.test_set.pad,
                                    end=self.test_set.eos, unk=self.test_set.unk)

        self.set_training(mode=True)

        if config.train_mode == 'gen':
            batchify = {
                'drectotde': [],
                'drecde': [],
                'dreccde': [],
                'dregde': [],
                'dhde': [],
            }
            with tqdm(loader) as pbar:
                for data in pbar:
                    data = self.preprocess_data(data)
                    bare, go, eos, sent_len = data

                    if config.model in ['vae', 'beta-vae']:
                        logits, pred_gaussian, init, last_states = self.AutoEncoder.saliency(bare=bare,
                                                                                             sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).sum(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian,
                                                                    standard_gaussian)  # shape = (n_batch, latent_dim)
                        kl_loss = kl_loss.sum(1).sum(0)

                        drecde = torch.autograd.grad(outputs=seq_loss, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        drecde = (drecde ** 2).sum(1)  # shape = (n_batch, )
                        drectotde = drecde  # shape = (n_batch, )
                        dreccde = torch.zeros_like(drecde)  # shape = (n_batch, )
                        dregde = torch.autograd.grad(outputs=kl_loss, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        dregde = (dregde ** 2).sum(1)  # shape = (n_batch, )
                    elif config.model in ['vae-couple', 'beta-vae-couple']:
                        logits, logits_couple, pred_gaussian, init, init_couple, last_states = self.AutoEncoder.saliency(
                            bare=bare,
                            sent_len=sent_len, go=go)

                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).sum(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, eos, keep_batch=True).sum(0)

                        standard_gaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                                                       gpu_wrapper(torch.ones(config.latent_dim)))
                        kl_loss = torch.distributions.kl_divergence(pred_gaussian,
                                                                    standard_gaussian)  # shape = (n_batch, latent_dim)
                        kl_loss = kl_loss.sum(1).sum(0)

                        drecde = torch.autograd.grad(outputs=seq_loss, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        drecde = (drecde ** 2).sum(1)  # shape = (n_batch, )
                        dreccde = torch.autograd.grad(outputs=seq_loss_couple, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        dreccde = (dreccde ** 2).sum(1)  # shape = (n_batch, )
                        drectotde = torch.autograd.grad(outputs=seq_loss + seq_loss_couple, inputs=last_states,
                                                      create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        drectotde = (drectotde ** 2).sum(1)  # shape = (n_batch, )
                        dregde = torch.autograd.grad(outputs=kl_loss, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        dregde = (dregde ** 2).sum(1)  # shape = (n_batch, )
                    elif config.model == 'dae':
                        logits, init, last_states = self.AutoEncoder.saliency(
                            bare=bare,
                            sent_len=sent_len, go=go)
                        seq_loss = self.criterionSeq(logits, eos, keep_batch=True).sum(0)
                        drecde = torch.autograd.grad(outputs=seq_loss, inputs=last_states,
                                                     create_graph=False, retain_graph=True, only_inputs=True)[
                            0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        drecde = (drecde ** 2).sum(1)  # shape = (n_batch, )
                        drectotde = drecde  # shape = (n_batch, )
                        dregde = torch.zeros_like(drecde)   # shape = (n_batch, )
                        dreccde = torch.zeros_like(drecde)   # shape = (n_batch, )
                    else:
                        raise ValueError()

                    dhde = 0
                    for _init in init.sum(0):
                        _dhde = torch.autograd.grad(outputs=_init, inputs=last_states,
                                                    create_graph=False, retain_graph=True, only_inputs=True)[0]  # shape = (n_batch, layers * n_dir * hid_dim)
                        _dhde = (_dhde ** 2).sum(1)  # shape = (n_batch, )
                        dhde = dhde + _dhde

                    seq_loss.backward()  # To free the GPU memory.

                    dhde = dhde / torch.sum(init ** 2, dim=1)  # shape = (n_batch, )

                    batchify['drectotde'].append(drectotde)
                    batchify['drecde'].append(drecde)
                    batchify['dreccde'].append(dreccde)
                    batchify['dregde'].append(dregde)
                    batchify['dhde'].append(dhde)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp

            drecde = batchify['drecde'].mean(0).item()
            dregde = batchify['dregde'].mean(0).item()
            drectotde = batchify['drectotde'].mean(0).item()
            dreccde = batchify['dreccde'].mean(0).item()
            dhde = batchify['dhde'].mean(0).item()
            print('\n\n\ndrecde = {}'.format(drecde))
            print('\n\n\ndregde = {}'.format(dregde))
            print('\n\n\ndreccde = {}'.format(dreccde))
            print('\n\n\ndrectotde = {}'.format(drectotde))
            print('\n\n\ndhde = {}'.format(dhde))
            print(config.save_model_dir)

        else:
            raise ValueError()

        self.set_training(mode=True)
        return

    def test_rec(self):
        if config.model in ['vae', 'vae-couple',
                            'vae-nf', 'vae-nf-couple',
                            'wae', 'wae-couple',
                            'wae-nf', 'wae-nf-couple',
                            'beta-vae', 'beta-vae-couple',
                            's-vae', 's-vae-couple',
                            'cnn-vae', 'cnn-vae-couple',
                            'cyc-anneal-vae', 'cyc-anneal-vae-couple',
                            'surprising-fix', 'surprising-fix-couple',
                            ]:
            pass
        elif config.model == 'lm':
            print('LM does not support reconstruction.')
        else:
            raise ValueError()

        dataset = self.test_set
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        n_sample = 10

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'recs': [],
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    data = self.preprocess_data(data)
                    bare, go, eos, sent_len = data

                    B = bare.shape[0]

                    sampled_latents = self.AutoEncoder.sample_from_posterior(bare=bare, sent_len=sent_len, n_sample=n_sample)
                    recs = self.AutoEncoder.decode_from(latents=sampled_latents.view(B * n_sample, config.latent_dim),
                                                        go=go.unsqueeze(1).expand(-1, n_sample, -1).contiguous().view(B * n_sample, -1))
                    recs = strip_eos([[self.vocab.id2word[idx.item()] for idx in rec] for rec in recs])

                    batchify['recs'].append(recs)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp
            ori_sents = dataset.get_sentences()

            log_dir = config.model_specific_dir('outputs/temp_results')
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            predictions = []
            references = []
            assert len(batchify['recs']) == len(ori_sents) * n_sample
            for sent_id in range(len(ori_sents)):
                for sample_id in range(n_sample):
                    predictions.append(' '.join(batchify['recs'][n_sample * sent_id + sample_id]))
                    references.append([' '.join(ori_sents[sent_id])])
            calc_bleu_score(
                predictions=predictions,
                references=references,
                log_dir=log_dir,
                multi_ref=True
            )
            print(config.save_model_dir)

            peep_num = 3
            for sent_id in range(peep_num):
                print('Input: {}'.format(' '.join(ori_sents[sent_id])))
                print()
                for sample_id in range(n_sample):
                    print('Rec{}: {}'.format(sample_id, ' '.join(batchify['recs'][n_sample * sent_id + sample_id])))
                print('-' * 50)
        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    def interpolation(self, beam_size=config.beam_size, save=False):
        if config.model == 'lm':
            print('LM does not support interpolation.')
            return

        if config.dataset == 'ptb':
            dataset = PTB_Interp()
        elif config.dataset == 'yelp':
            dataset = Yelp_Interp()
        elif config.dataset == 'yahoo':
            dataset = Yahoo_Interp()
        else:
            raise ValueError()

        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'sentA': [],
                'sentB': [],
                'interps': [],
            }
            top_three = [False, True][1]
            c = 0
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    if top_three:
                        c += 1
                        if c > 3:
                            break
                    data = self.preprocess_data(data)
                    bareA, goA, _, sent_lenA, bareB, _, _, sent_lenB = data

                    n_interps = 5
                    interps = self.AutoEncoder.gen_interps(bareA=bareA, sent_lenA=sent_lenA,
                                                           bareB=bareB, sent_lenB=sent_lenB,
                                                           go=goA, n_interps=n_interps)
                    # interps: [n_batch x [n_interps x preds-like list]]

                    batchify['sentA'].append(bareA)
                    batchify['sentB'].append(bareB)
                    batchify['interps'].append(interps)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data  # shape = (n_tot, ?)
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp
            batchify['sentA'] = [strip_pad([self.vocab.id2word[idx.item()] for idx in sentA]) for sentA in batchify['sentA']]
            batchify['sentB'] = [strip_pad([self.vocab.id2word[idx.item()] for idx in sentB]) for sentB in batchify['sentB']]
            batchify['interps'] = [[strip_eos([self.vocab.id2word[idx.item()] for idx in interp]) for interp in interps] for interps in batchify['interps']]

            if True:

                # ----- Peep the resutls -----
                peeps = 10
                peepe = 20
                for sentA, sentB, interps in zip(batchify['sentA'][peeps:peepe],
                                                 batchify['sentB'][peeps:peepe],
                                                 batchify['interps'][peeps:peepe]):
                    print('sentA              = ' + ' '.join(sentA))
                    for in_id, interp in enumerate(interps):
                        print('interp{}            = '.format(in_id) + ' '.join(interp))
                    print('sentB              = ' + ' '.join(sentB))
                    print('-' * 50)
                for idx in range(3):
                    print('*' * 150)

                # ----- Save the results to file -----
                save_num = 40
                if save:
                    save_file = os.path.join(config.sample_dir, 'interpolation.txt')
                    if os.path.exists(save_file):
                        print('WARNING. The file interpolation.txt exists.')
                    with open(save_file, 'w') as f:
                        for sentA, sentB, interps in zip(batchify['sentA'][:save_num],
                                                         batchify['sentB'][:save_num],
                                                         batchify['interps'][:save_num]):
                            f.write('sentA              = ' + ' '.join(sentA) + '\n')
                            for in_id, interp in enumerate(interps):
                                f.write('interp{}            = '.format(in_id) + ' '.join(interp) + '\n')
                            f.write('sentB              = ' + ' '.join(sentB) + '\n')
                            f.write('-' * 50 + '\n')
        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    @staticmethod
    def preprocess_data(data):
        return [gpu_wrapper(item) for item in data]


def compute_mmd(posterior_samples):

    B = posterior_samples.shape[0]
    prior_dist = torch.distributions.Normal(gpu_wrapper(torch.zeros(config.latent_dim)),
                                            gpu_wrapper(torch.ones(config.latent_dim)))
    prior_samples = prior_dist.sample(torch.Size([B]))  # shape = (n_batch, latent_dim)

    norms_prior = torch.sum(prior_samples.pow(2), dim=1, keepdim=True)  # shape = (n_batch, 1)
    dotprobs_prior = torch.matmul(prior_samples, prior_samples.t())  # shape = (n_batch, n_batch)
    distances_prior = norms_prior + norms_prior.t() - 2. * dotprobs_prior  # shape = (n_batch, n_batch)

    norms_posterior = torch.sum(posterior_samples.pow(2), dim=1, keepdim=True)  # shape = (n_batch, 1)
    dotprobs_posterior = torch.matmul(posterior_samples, posterior_samples.t())  # shape = (n_batch, n_batch)
    distances_posterior = norms_posterior + norms_posterior.t() - 2. * dotprobs_posterior  # shape = (n_batch, n_batch)

    dotprobs = torch.matmul(posterior_samples, prior_samples.t())  # shape = (n_batch, n_batch)
    distances = norms_posterior + norms_prior.t() - 2. * dotprobs  # shape = (n_batch, n_batch)

    if config.mmd_kernel == 'rbf':
        raise NotImplementedError()
        # Median heuristic for the sigma^2 of Gaussian kernel
        half_size = (n * n - n) / 2.0
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        res1 = tf.exp(- distances_qz / 2. / sigma2_k)
        res1 += tf.exp(- distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp(- distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif config.mmd_kernel == 'imq':
        # k(x, y) = C / (C + ||x - y||^2)
        Cbase = 2. * config.latent_dim * 1.  # For sigma2_p = 1.
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_posterior)  # shape = (n_batch, n_batch)
            res1 += C / (C + distances_prior)  # shape = (n_batch, n_batch)

            res1 = res1 * (1 - gpu_wrapper(torch.eye(B)))  # shape = (n_batch, n_batch)
            res1 = torch.sum(res1) / (B * B - B)
            res2 = C / (C + distances)  # shape = (n_batch, n_batch)
            res2 = torch.sum(res2) * 2. / (B * B)  # shape = (n_batch, n_batch)
            stat += (res1 - res2)
    else:
        raise ValueError()

    return stat


def compute_imq_dist_couple(euc_dist):
    # k(x, y) = C / (C + ||x - y||^2)
    Cbase = 2. * config.latent_dim  # For sigma2_p = 1.
    imq = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        imq = imq + C / (C + euc_dist)  # shape = (n_batch, n_batch)
    return -1 * imq


def compute_imq_dist(euc_dist):
    # k(x, y) = C / (C + ||x - y||^2)
    Cbase = 2. * config.latent_dim * config.fgm_epsilon * config.ita_c  # For sigma2_p = 1.
    imq = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        imq = imq + C / (C + euc_dist)  # shape = (n_batch, n_batch)
    return -1 * imq


def euclidean(a, b, clip=None):
    """

    :param a: shape = (n_batch, latent_dim)
    :param b: shape = (n_batch, latent_dim)
    :param clip: float.
    :return:
    """
    if clip is not None:
        return torch.clamp(((a - b) ** 2).sum(-1), max=clip)
    else:
        return ((a - b) ** 2).sum(-1)


def cos(a, b):
    """
    Both a and b are normalized.
    :param a: shape = (n_batch, latent_dim)
    :param b: shape = (n_batch, latent_dim)
    :return:
    """
    # a = a / a.norm(dim=-1, keepdim=True)
    # b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(1)


class MyMesh(object):

    def __init__(self,
                 height,
                 width,
                 n_rows,
                 n_cols,
                 ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig = plt.figure(figsize=(height, width))
        self.axes = self.fig.subplots(n_rows, n_cols, squeeze=False)

    def get_axis(self, row, col):
        assert (0 <= row < self.n_rows) and (0 <= col < self.n_cols)
        return self.axes[row, col]

    def save_fig(self, save_name):
        self.fig.savefig(save_name)