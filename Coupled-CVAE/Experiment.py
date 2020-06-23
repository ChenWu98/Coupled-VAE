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
from Modules.seq2seq import SEQ2SEQ
from Modules.vae import VAE
from Modules.vae_couple import VAE_COUPLE
from Modules.Losses.SeqLoss import SeqLoss
from Modules.Losses.Reward import RewardCriterion
from Modules.Losses.GANLoss import GANLoss
from tqdm import tqdm
from dataloaders.switchboard import SwitchBoard
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
        if config.dataset == 'switchboard':
            if not self.is_test:
                self.train_set = SwitchBoard('train')
            self.test_set = SwitchBoard('test')
            self.val_set = SwitchBoard('valid')
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

        if config.model == 'seq2seq':
            self.AutoEncoder = SEQ2SEQ(hid_dim=config.hid_dim,
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
        elif config.model == 'cvae':
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
        elif config.model == 'cvae-couple':
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
        else:
            raise ValueError()

        if config.model in ['seq2seq',
                            'cvae', 'cvae-couple',
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
            if config.model in ['seq2seq',
                                'cvae', 'cvae-couple',
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
                                  'cvae', 'cvae-couple',
                                  'vae-nf', 'vae-nf-couple',
                                  's-vae', 's-vae-couple',
                                  'cnn-vae', 'cnn-vae-couple',
                                  ]:
                max_val = 1
            elif config.model in ['wae', 'wae-couple',
                                  'wae-nf', 'wae-nf-couple',
                                  ]:
                max_val = 0.8
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
        # self.evaluate_dialogue('test')
        # self.evaluate_diversity('test', beam_size=1, save=True)
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
        self.evaluate_dialogue(val_or_test='test', save=True)
        # Sample from prior test.
        # for beam_size in config.enum_beam_size:
        #     self.evaluate_diversity(beam_size=beam_size, save=True)
        # if config.model != 'seq2seq':
        #     self.evaluate_diversity('test', beam_size=1, save=True)
        self.evaluate_diversity('test', beam_size=1, save=True)

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
                    post_bare, post_len, resp_bare, resp_go, resp_eos, resp_len = data

                    max_len = max(post_len).item()
                    post_bare = post_bare[:, :max_len].contiguous()

                    max_len = max(resp_len).item()
                    resp_bare = resp_bare[:, :max_len].contiguous()
                    resp_go = resp_go[:, :max_len + 1].contiguous()
                    resp_eos = resp_eos[:, :max_len + 1].contiguous()

                    if config.model == 'seq2seq':
                        logits = self.AutoEncoder(
                            post_bare=post_bare,  # shape = (n_batch, 15)
                            post_len=post_len,  # shape = (n_batch, )
                            resp_go=resp_go,  # shape = (n_batch, 16)
                            resp_len=resp_len,  # shape = (n_batch, )
                            resp_bare=resp_bare  # shape = (n_batch, 15)
                        )  # shape = (n_batch, 16, V)
                        seq_loss = self.criterionSeq(logits, resp_eos, keep_batch=True).mean(0)
                        tot_loss = seq_loss

                        # ----- Logging -----
                        loss['seq/L'] = round(seq_loss.item(), ROUND)

                        # ----- Backward for scopes: ['gen'] -----
                        self.zero_grad()
                        tot_loss.backward()
                        self.step(['gen'])

                    elif config.model in ['cvae']:
                        logits, prior_dist, posterior_dist = self.AutoEncoder(
                            post_bare=post_bare,  # shape = (n_batch, 15)
                            post_len=post_len,  # shape = (n_batch, )
                            resp_go=resp_go,  # shape = (n_batch, 16)
                            resp_len=resp_len,  # shape = (n_batch, )
                            resp_bare=resp_bare  # shape = (n_batch, 15)
                        )

                        seq_loss = self.criterionSeq(logits, resp_eos, keep_batch=True).mean(0)
                        kl_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist)  # shape = (n_batch, emb_dim)
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

                    elif config.model in ['cvae-couple']:
                        logits, prior_dist, posterior_dist, logits_couple, init, init_couple = self.AutoEncoder(
                            post_bare=post_bare,  # shape = (n_batch, 15)
                            post_len=post_len,  # shape = (n_batch, )
                            resp_go=resp_go,  # shape = (n_batch, 16)
                            resp_len=resp_len,  # shape = (n_batch, )
                            resp_bare=resp_bare  # shape = (n_batch, 15)
                        )

                        seq_loss = self.criterionSeq(logits, resp_eos, keep_batch=True).mean(0)
                        seq_loss_couple = self.criterionSeq(logits_couple, resp_eos, keep_batch=True).mean(0)

                        kl_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist)  # shape = (n_batch, emb_dim)
                        kl_loss = kl_loss.sum(1).mean(0)

                        euc_init = euclidean(init, init_couple.detach(), clip=None)  # shape = (n_batch, )

                        imq_kernel_dist = compute_imq_dist_couple(euc_init)  # shape = (n_batch, )
                        if config.model == 'cvae-couple' and config.euclidean_dist:
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
                    # self.evaluate_diversity()
                    self.evaluate_dialogue('val')
                    # self.evaluate_diversity('val', beam_size=1, save=True)

                # Decay learning rates.
                if self.iter_num % config.lr_decay_step == 0 and self.iter_num > config.start_decay:
                    self.update_lr()

    def evaluate_diversity(self, val_or_test, beam_size=config.beam_size, save=False):
        # if config.model == 'seq2seq':
        #     raise ValueError()

        self.AutoEncoder.Decoder.beam_size = beam_size
        n_sample = 100
        dataset = {
            "test": self.test_set,
            "val": self.val_set
        }[val_or_test]
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

        self.set_training(mode=False)

        if config.train_mode == 'gen':
            batchify = {
                'preds': [],
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    data = self.preprocess_data(data)
                    post_bare, post_len, resp_bare, resp_go, resp_eos, resp_len = data

                    max_len = max(post_len).item()
                    post_bare = post_bare[:, :max_len].contiguous()

                    max_len = max(resp_len).item()
                    resp_bare = resp_bare[:, :max_len].contiguous()
                    resp_go = resp_go[:, :max_len + 1].contiguous()
                    resp_eos = resp_eos[:, :max_len + 1].contiguous()

                    preds = self.AutoEncoder.sample_from_prior(
                            post_bare=post_bare,
                            post_len=post_len,
                            resp_go=resp_go,
                        )
                    batchify['preds'].append(preds)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data
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

        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    def evaluate_dialogue(self, val_or_test, save=False):
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
                'resp_len': []
            }
            with tqdm(loader) as pbar, torch.no_grad():
                for data in pbar:
                    data = self.preprocess_data(data)
                    post_bare, post_len, resp_bare, resp_go, resp_eos, resp_len = data

                    max_len = max(post_len).item()
                    post_bare = post_bare[:, :max_len].contiguous()

                    max_len = max(resp_len).item()
                    resp_bare = resp_bare[:, :max_len].contiguous()
                    resp_go = resp_go[:, :max_len + 1].contiguous()
                    resp_eos = resp_eos[:, :max_len + 1].contiguous()

                    xent, nll, kl = self.AutoEncoder.test_lm(
                        post_bare=post_bare,
                        post_len=post_len,
                        resp_go=resp_go,
                        resp_len=resp_len,
                        resp_bare=resp_bare,
                        resp_eos=resp_eos,
                        n_sample=n_sample
                    )

                    batchify['xent'].append(xent)
                    batchify['nll'].append(nll)
                    batchify['kl'].append(kl)
                    batchify['resp_len'].append(resp_len)

            # ----- De-batchify -----
            for key in batchify.keys():
                if len(batchify[key]) > 0 and isinstance(batchify[key][0], torch.Tensor):
                    batchify[key] = torch.cat(batchify[key], dim=0).cpu().data
                elif len(batchify[key]) > 0 and isinstance(batchify[key][0], list):
                    temp = []
                    for batch in batchify[key]:
                        temp.extend(batch)
                    batchify[key] = temp
            # batchify['preds'] = [strip_eos([self.vocab.id2word[idx.item()] for idx in pred]) for pred in batchify['preds']]

            # ---- Get original articles and references -----
            posts, responses = dataset.get_sentences()

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

            ppl = torch.exp(batchify['nll'].sum(0) / (batchify['resp_len'] + 1).float().sum(0)).item()
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
                for xent, nll, kl, post, resp in zip(batchify['xent'][:peep],
                                                     batchify['nll'][:peep],
                                                     batchify['kl'][:peep],
                                                     posts[:peep],
                                                     responses[:peep]):
                    print('POST              = ' + ' '.join(post))
                    print('RESP              = ' + ' '.join(resp))
                    print('NLL               = {}'.format(nll.item()))
                    print('XENT              = {}'.format(xent.item()))
                    print('KL                = {}'.format(kl.item()))
                    print('-' * 50)
                for idx in range(3):
                    print('*' * 150)
        else:
            raise ValueError()

        self.set_training(mode=True)
        return None

    @staticmethod
    def preprocess_data(data):
        return [gpu_wrapper(item) for item in data]


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