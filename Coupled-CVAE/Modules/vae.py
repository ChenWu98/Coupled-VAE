import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper, gumbel_softmax, log_sum_exp
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from Modules.encoder import Encoder
from Modules.decoder import Decoder
from Modules.Losses.SeqLoss import SeqLoss
from Modules.vae_couple import _n_sample


class VAE(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(VAE, self).__init__()
        assert encoder_emb.num_embeddings == decoder_emb.num_embeddings
        assert encoder_emb.embedding_dim == decoder_emb.embedding_dim
        self.voc_size = encoder_emb.num_embeddings
        self.emb_dim = encoder_emb.embedding_dim
        self.hid_dim = hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.enc_bi = enc_bi
        self.n_dir = 2 if self.enc_bi else 1
        self.dec_max_len = dec_max_len
        self.beam_size = beam_size
        self.WEAtt_type = WEAtt_type
        self.latent_dim = latent_dim

        self.PostEncoder = Encoder(emb_dim=self.emb_dim,
                                   hid_dim=self.hid_dim,
                                   n_layer=self.enc_layers,
                                   dropout=self.dropout,
                                   bi=self.enc_bi,
                                   embedding=encoder_emb)
        self.RespEncoder = Encoder(emb_dim=self.emb_dim,
                                   hid_dim=self.hid_dim,
                                   n_layer=self.enc_layers,
                                   dropout=self.dropout,
                                   bi=self.enc_bi,
                                   embedding=encoder_emb)
        self.PriorGaussian = Gaussian(in_dim=self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim)
        self.PosteriorGaussian = Gaussian(in_dim=2 * self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim)
        self.PostRepr = nn.Linear(self.hid_dim * self.n_dir * self.enc_layers, self.emb_dim)
        self.Decoder = Decoder(voc_size=self.voc_size,
                               latent_dim=self.latent_dim,
                               emb_dim=self.emb_dim,
                               hid_dim=self.hid_dim * self.n_dir,
                               n_layer=self.dec_layers,
                               dropout=self.dropout,
                               max_len=self.dec_max_len,
                               beam_size=self.beam_size,
                               WEAtt_type=self.WEAtt_type,
                               embedding=decoder_emb)
        self.BoW = nn.Linear(self.latent_dim, self.voc_size)

        self.criterionSeq = SeqLoss(voc_size=self.voc_size, pad=pad_id, end=None, unk=None)

    def test_lm(self, post_bare, post_len, resp_go, resp_len, resp_bare, resp_eos, n_sample):
        B = post_bare.shape[0]

        # ----- Post Encoding -----
        post_outputs, post_last_states = self.PostEncoder(post_bare, post_len)
        # post_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # post_last_states.shape = (layers * n_dir, n_batch, hid_dim)
        post_last_states = post_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)
        post_repr = self.PostRepr(post_last_states)  # shape = (n_batch, emb_dim)

        # ----- Response Encoding -----
        _, resp_last_states = self.RespEncoder(resp_bare, resp_len)
        # resp_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # resp_last_states.shape = (layers * n_dir, n_batch, hid_dim)
        resp_last_states = resp_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Prior Network -----
        prior_dist, prior_latent = self.PriorGaussian(post_last_states)
        # prior_latent.shape = (n_batch, hid_dim)

        # ----- Posterior Network -----
        posterior_dist, posterior_latent = self.PosteriorGaussian(torch.cat([resp_last_states, post_last_states], dim=1))
        # posterior_latent.shape = (n_batch, hid_dim)

        # ----- Importance sampling estimation -----
        xent, nll, kl = self.importance_sampling(prior_dist=prior_dist,
                                                 posterior_dist=posterior_dist,
                                                 resp_go=resp_go,
                                                 resp_eos=resp_eos,
                                                 post_repr=post_repr,
                                                 n_sample=n_sample)
        # xent.shape = (n_batch, )
        # nll.shape = (n_batch, )
        # kl.shape = (n_batch, )

        return xent, nll, kl

    def importance_sampling(self, prior_dist, posterior_dist, resp_go, resp_eos, post_repr, n_sample):
        B = resp_go.shape[0]
        assert n_sample % _n_sample == 0

        samplify = {
            'xent': [],
            'log_pz': [],
            'log_pxz': [],
            'log_qzx': [],
            'z': []
        }
        for sample_id in range(n_sample // _n_sample):

            # ----- Sampling -----
            _z = posterior_dist.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, latent_dim)
            assert tuple(_z.shape) == (_n_sample, B, self.latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            _init_states = gpu_wrapper(torch.zeros([self.enc_layers, _n_sample * B, self.n_dir * self.hid_dim])).float()  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)

            # ----- Importance sampling for NLL -----
            _logits = self.Decoder(init_states=_init_states,  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)
                                   post_repr=post_repr.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, emb_dim)
                                   latent_vector=_z.contiguous().view(_n_sample * B, self.latent_dim),  # shape = (_n_sample * n_batch, out_dim)
                                   helper=resp_go.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 15)
                                   test_lm=True)  # shape = (_n_sample * n_batch, 16, V)
            _xent = self.criterionSeq(_logits,  # shape = (_n_sample * n_batch, 16, V)
                                      resp_eos.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 16)
                                      keep_batch=True).view(_n_sample, B)  # shape = (_n_sample, n_batch)

            _log_pz = prior_dist.log_prob(_z).sum(2)  # shape = (_n_sample, n_batch)
            _log_pxz = - _xent  # shape = (_n_sample, n_batch)
            _log_qzx = posterior_dist.log_prob(_z).sum(2)  # shape = (_n_sample, n_batch)

            samplify['xent'].append(_xent)  # shape = (_n_sample, n_batch)
            samplify['log_pz'].append(_log_pz)  # shape = (_n_sample, n_batch)
            samplify['log_pxz'].append(_log_pxz)  # shape = (_n_sample, n_batch)
            samplify['log_qzx'].append(_log_qzx)  # shape = (_n_sample, n_batch)
            samplify['z'].append(_z)  # shape = (_n_sample, n_batch, out_dim)

        for key in samplify.keys():
            samplify[key] = torch.cat(samplify[key], dim=0)  # shape = (n_sample, ?)

        ll = log_sum_exp(samplify['log_pz'] + samplify['log_pxz'] - samplify['log_qzx'], dim=0) - np.log(n_sample)  # shape = (n_batch, )
        nll = - ll  # shape = (n_batch, )

        # ----- Importance sampling for KL -----
        # kl = kl_with_isogaussian(gaussian_dist)  # shape = (n_batch, )
        kl = (samplify['log_qzx'] - samplify['log_pz']).mean(0)  # shape = (n_batch, )

        return samplify['xent'].mean(0), nll, kl

    def generate_gaussian(self, B):
        return self.PriorGaussian.sample(torch.Size([B]))  # shape = (n_batch, emb_dim)

    def sample_from_prior(self, post_bare, post_len, resp_go):
        """

        :param go: shape = (n_batch, 16)
        :return:
        """
        B = resp_go.shape[0]

        # ----- Post Encoding -----
        post_outputs, post_last_states = self.PostEncoder(post_bare, post_len)
        # post_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # post_last_states.shape = (layers * n_dir, n_batch, hid_dim)
        post_last_states = post_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)
        post_repr = self.PostRepr(post_last_states)  # shape = (n_batch, emb_dim)

        # ----- Prior Network -----
        prior_dist, prior_latent = self.PriorGaussian(post_last_states)
        # prior_latent.shape = (n_batch, hid_dim)

        # ----- Initial Decoding States -----
        assert self.enc_bi
        init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        preds = self.Decoder(init_states=init_states,
                             post_repr=post_repr,
                             latent_vector=prior_latent,
                             helper=resp_go)
        return preds

    def forward(self, post_bare, post_len, resp_go, resp_len, resp_bare):
        """

        :param post_bare: shape = (n_batch, 15)
        :param post_len: shape = (n_batch, )
        :param resp_go: shape = (n_batch, 16)
        :param resp_len: shape = (n_batch, 15)
        :param resp_bare: shape = (n_batch, 15)
        :return:
        """
        B = resp_go.shape[0]

        if not self.training:
            raise NotImplementedError()
        else:
            # ----- Post Encoding -----
            post_outputs, post_last_states = self.PostEncoder(post_bare, post_len)
            # post_outputs.shape = (n_batch, 15, n_dir * hid_dim)
            # post_last_states.shape = (layers * n_dir, n_batch, hid_dim)
            post_last_states = post_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)
            post_repr = self.PostRepr(post_last_states)  # shape = (n_batch, emb_dim)

            # ----- Response Encoding -----
            _, resp_last_states = self.RespEncoder(resp_bare, resp_len)
            # resp_outputs.shape = (n_batch, 15, n_dir * hid_dim)
            # resp_last_states.shape = (layers * n_dir, n_batch, hid_dim)
            resp_last_states = resp_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

            # ----- Prior Network -----
            prior_dist, prior_latent = self.PriorGaussian(post_last_states)
            # prior_latent.shape = (n_batch, hid_dim)

            # ----- Posterior Network -----
            posterior_dist, posterior_latent = self.PosteriorGaussian(torch.cat([resp_last_states, post_last_states], dim=1))
            # posterior_latent.shape = (n_batch, hid_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                post_repr=post_repr,
                                latent_vector=posterior_latent,
                                helper=resp_go), prior_dist, posterior_dist


class Gaussian(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Gaussian, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.toMiu = nn.Sequential(nn.Linear(self.in_dim, self.out_dim))
        self.toSigma = nn.Sequential(nn.Linear(self.in_dim, self.out_dim))

    def forward(self, input):
        """

        :param input: shape = (n_batch, in_dim)
        :return: gaussian_dist is a Distribution object and latent_vector.shape = (n_batch, out_dim)
        """

        miu = self.toMiu(input)  # shape = (n_batch, out_dim)
        sigma = torch.exp(self.toSigma(input))  # shape = (n_batch, out_dim)
        gaussian_dist = torch.distributions.Normal(miu, sigma)
        latent_vector = gaussian_dist.rsample(torch.Size([1])).squeeze(0)  # shape = (1, n_batch, out_dim) -> (n_batch, out_dim)

        return gaussian_dist, latent_vector
