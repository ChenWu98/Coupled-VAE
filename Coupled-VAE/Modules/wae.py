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
from Modules.vae_nf_geo import _n_sample


def kl_with_isogaussian(gaussian_dist):

    var_ratio = (gaussian_dist.scale / 1).pow(2)
    t1 = ((gaussian_dist.loc - 0) / 1).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log()).sum(-1)


class WAE(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(WAE, self).__init__()
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

        self.Encoder = Encoder(emb_dim=self.emb_dim,
                               hid_dim=self.hid_dim,
                               n_layer=self.enc_layers,
                               dropout=self.dropout,
                               bi=self.enc_bi,
                               embedding=encoder_emb)
        self.PriorGaussian = torch.distributions.Normal(gpu_wrapper(torch.zeros(self.latent_dim)),
                                                        gpu_wrapper(torch.ones(self.latent_dim)))
        self.PosteriorGaussian = Gaussian(in_dim=self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim)
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
        self.criterionSeq = SeqLoss(voc_size=self.voc_size, pad=pad_id, end=None, unk=None)

    def estimate_mi(self, go, sent_len, bare, n_sample):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        gaussian_dist, _ = self.PosteriorGaussian(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        mi, sampled_latents = self.importance_sampling_mi(gaussian_dist=gaussian_dist, n_sample=n_sample)  # shape = (n_batch, )

        return mi, sampled_latents

    def importance_sampling_mi(self, gaussian_dist, n_sample):
        assert n_sample % _n_sample == 0

        B = gaussian_dist.mean.shape[0]

        samplify = {
            'log_qz': [],
            'log_qzx': [],
            'z': []
        }
        for sample_id in range(n_sample // _n_sample):
            # ----- Sampling -----
            _z = gaussian_dist.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, latent_dim)
            assert tuple(_z.shape) == (_n_sample, B, self.latent_dim)

            _log_qzx = gaussian_dist.log_prob(_z).sum(2)  # shape = (_n_sample, n_batch)
            _log_qz = gaussian_dist.log_prob(_z.unsqueeze(2).expand(-1, -1, B, -1)).sum(3)  # shape = (_n_sample, n_batch, n_batch)
            # Exclude itself.
            _log_qz.masked_fill_(gpu_wrapper(torch.eye(B).long()).eq(1).unsqueeze(0).expand(_n_sample, -1, -1), -float('inf'))  # shape = (_n_sample, n_batch, n_batch)
            _log_qz = (log_sum_exp(_log_qz, dim=2) - np.log(B - 1))  # shape = (_n_sample, n_batch)

            samplify['log_qzx'].append(_log_qzx)  # shape = (_n_sample, n_batch)
            samplify['log_qz'].append(_log_qz)  # shape = (_n_sample, n_batch)
            samplify['z'].append(_z)  # shape = (_n_sample, n_batch, out_dim)

        for key in samplify.keys():
            samplify[key] = torch.cat(samplify[key], dim=0)  # shape = (n_sample, ?)

        # ----- Importance sampling for MI -----
        mi = samplify['log_qzx'].mean(0) - samplify['log_qz'].mean(0)

        return mi, samplify['z'].transpose(0, 1)

    def test_lm(self, go, sent_len, bare, eos, n_sample):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        gaussian_dist, _ = self.PosteriorGaussian(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        xent, nll, kl, sampled_latents = self.importance_sampling(gaussian_dist=gaussian_dist, go=go, eos=eos, n_sample=n_sample)
        # xent.shape = (n_batch, )
        # nll.shape = (n_batch, )
        # kl.shape = (n_batch, )
        # sampled_latents.shape = (n_batch, n_sample, out_dim)

        return xent, nll, kl, sampled_latents

    def importance_sampling(self, gaussian_dist, go, eos, n_sample):
        B = go.shape[0]
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
            _z = gaussian_dist.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, out_dim)
            assert tuple(_z.shape) == (_n_sample, B, self.latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            _init_states = gpu_wrapper(torch.zeros([self.enc_layers, _n_sample * B, self.n_dir * self.hid_dim])).float()  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)

            # ----- Importance sampling for NLL -----
            _logits = self.Decoder(init_states=_init_states,  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)
                                   latent_vector=_z.contiguous().view(_n_sample * B, self.latent_dim),  # shape = (_n_sample * n_batch, out_dim)
                                   helper=go.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 15)
                                   test_lm=True)  # shape = (_n_sample * n_batch, 16, V)
            _xent = self.criterionSeq(_logits,  # shape = (_n_sample * n_batch, 16, V)
                                      eos.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 16)
                                      keep_batch=True).view(_n_sample, B)  # shape = (_n_sample, n_batch)

            _log_pz = self.PriorGaussian.log_prob(_z).sum(2)  # shape = (_n_sample, n_batch)
            _log_pxz = - _xent  # shape = (_n_sample, n_batch)
            _log_qzx = gaussian_dist.log_prob(_z).sum(2)  # shape = (_n_sample, n_batch)

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

        return samplify['xent'].mean(0), nll, kl, samplify['z'].transpose(0, 1)

    def generate_gaussian(self, B):
        return self.PriorGaussian.sample(torch.Size([B]))  # shape = (n_batch, emb_dim)

    def sample_from_prior(self):
        raise NotImplementedError()

    def sample_from_posterior(self, bare, sent_len, n_sample):
        """

        :param bare: shape = (n_batch, 15)
        :param sent_len: shape = (n_batch, )
        :param n_sample: int
        :return: shape = (n_batch, n_samples, latent_dim)
        """

        B = bare.shape[0]
        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        gaussian_dist, _ = self.PosteriorGaussian(last_states)

        samples = gaussian_dist.sample(torch.Size([n_sample]))  # shape = (n_sample, n_batch, latent_dim)
        samples = samples.transpose(0, 1).contiguous()  # shape = (n_batch, n_sample, latent_dim)

        return samples

    def decode_from(self, latents, go):
        """

        :param latents: shape = (n_batch, latent_dim)
        :param go: shape = (n_batch, 16)
        :return:
        """
        B = latents.shape[0]

        init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        return self.Decoder(init_states=init_states,
                            latent_vector=latents,
                            helper=go)

    def forward(self, go, sent_len=None, bare=None):
        """

        :param go: shape = (n_batch, 16)
        :param sent_len: shape = (n_batch, ) or None
        :param bare: shape = (n_batch, 15) or None
        :return:
        """
        B = go.shape[0]

        if not self.training:
            # ----- Prior Network -----
            latent_vector = self.generate_gaussian(B)  # shape = (n_batch, latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                latent_vector=latent_vector,
                                helper=go)
        else:
            # ----- Encoding -----
            outputs, last_states = self.Encoder(bare, sent_len)
            # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
            # last_states.shape = (layers * n_dir, n_batch, hid_dim)
            last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim

            # ----- Posterior Network -----
            gaussian_dist, latent_vector = self.PosteriorGaussian(last_states)
            # latent_vector.shape = (n_batch, latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                latent_vector=latent_vector,
                                helper=go), gaussian_dist, latent_vector


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
