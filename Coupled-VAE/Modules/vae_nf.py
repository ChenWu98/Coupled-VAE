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
from Modules.subModules.flows import NormalizingFlows
from Modules.vae_nf_geo import _n_sample


class VAE_NF(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id, n_flows, flow_type):
        super(VAE_NF, self).__init__()
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
        self.n_flows = n_flows
        self.flow_type = flow_type

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
        self.BoW = nn.Linear(self.latent_dim, self.voc_size)
        self.Flows = NormalizingFlows(cond_dim=self.hid_dim * self.n_dir * self.enc_layers,
                                      latent_dim=self.latent_dim,
                                      n_flows=self.n_flows,
                                      flow_type=self.flow_type)

        self.criterionSeq = SeqLoss(voc_size=self.voc_size, pad=pad_id, end=None, unk=None)

    def estimate_mi(self, go, sent_len, bare, n_sample):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        Q0, _ = self.PosteriorGaussian(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        mi, sampled_latents = self.importance_sampling_mi(Q0=Q0, last_states=last_states, n_sample=n_sample)  # shape = (n_batch, )

        return mi, sampled_latents

    def importance_sampling_mi(self, Q0, last_states, n_sample):
        assert n_sample % _n_sample == 0

        B = Q0.mean.shape[0]

        samplify = {
            'log_qz': [],
            'log_qzx': [],
            'z': []
        }
        for sample_id in range(n_sample // _n_sample):
            # ----- Sampling -----
            _z0 = Q0.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, out_dim)
            assert tuple(_z0.shape) == (_n_sample, B, self.latent_dim)

            # ----- Flows -----
            _zk, _sum_log_jacobian = self.Flows(z0=_z0.contiguous().view(_n_sample * B, self.latent_dim), # shape = (_n_sample * n_batch, out_dim)
                                                cond=last_states.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1)
                                                # shape = (_n_sample * n_batch, layers * n_dir * hid_dim)
                                                )
            # _zk.shape = (_n_sample * n_batch, latent_dim)
            # _sum_log_jacobian.shape = (_n_sample * n_batch, )
            _zk = _zk.view(_n_sample, B, self.latent_dim)  # shape = (_n_sample, n_batch, latent_dim)
            _sum_log_jacobian = _sum_log_jacobian.view(_n_sample, B)  # shape = (_n_sample, n_batch)

            # ----- Flows for the aggregate posterior -----
            _, _sum_log_jacobian_batch = self.Flows(z0=_z0.unsqueeze(2).expand(-1, -1, B, -1).contiguous().view(_n_sample * B * B, self.latent_dim),  # shape = (_n_sample * n_batch * n_batch, out_dim)
                                                    cond=last_states.unsqueeze(0).unsqueeze(1).expand(_n_sample, B, -1, -1).contiguous().view(_n_sample * B * B, -1)  # shape = (_n_sample * n_batch * n_batch, layers * n_dir * hid_dim)
                                                    )
            # _sum_log_jacobian_batch.shape = (_n_sample * n_batch * n_batch, )
            _sum_log_jacobian_batch = _sum_log_jacobian_batch.view(_n_sample, B, B)  # shape = (_n_sample, n_batch, n_batch)

            _log_qzx = Q0.log_prob(_z0).sum(2) - _sum_log_jacobian  # shape = (_n_sample, n_batch)
            _log_qz = Q0.log_prob(_z0.unsqueeze(2).expand(-1, -1, B, -1)).sum(3) - _sum_log_jacobian_batch  # shape = (_n_sample, n_batch, n_batch)
            # Exclude itself.
            _log_qz.masked_fill_(gpu_wrapper(torch.eye(B).long()).eq(1).unsqueeze(0).expand(_n_sample, -1, -1), -float('inf'))  # shape = (_n_sample, n_batch, n_batch)
            _log_qz = (log_sum_exp(_log_qz, dim=2) - np.log(B - 1))  # shape = (_n_sample, n_batch)

            samplify['log_qzx'].append(_log_qzx)  # shape = (_n_sample, n_batch)
            samplify['log_qz'].append(_log_qz)  # shape = (_n_sample, n_batch)
            samplify['z'].append(_zk)  # shape = (_n_sample, n_batch, out_dim)

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
        Q0, _ = self.PosteriorGaussian(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        xent, nll, kl, sampled_latents = self.importance_sampling(Q0=Q0, go=go, eos=eos, last_states=last_states, n_sample=n_sample)
        # xent.shape = (n_batch, )
        # nll.shape = (n_batch, )
        # kl.shape = (n_batch, )
        # sampled_latents.shape = (n_batch, n_sample, out_dim)

        return xent, nll, kl, sampled_latents

    def importance_sampling(self, Q0, go, eos, last_states, n_sample):
        B = go.shape[0]
        assert n_sample % _n_sample == 0

        samplify = {
            'xent': [],
            'log_pzk': [],
            'log_pxzk': [],
            'log_qzkx': [],
            'zk': []
        }
        for sample_id in range(n_sample // _n_sample):

            # ----- Sampling -----
            _z0 = Q0.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, out_dim)
            assert tuple(_z0.shape) == (_n_sample, B, self.latent_dim)

            # ----- Flows -----
            _zk, _sum_log_jacobian = self.Flows(z0=_z0.contiguous().view(_n_sample * B, self.latent_dim),  # shape = (_n_sample * n_batch, out_dim)
                                                cond=last_states.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1)  # shape = (_n_sample * n_batch, layers * n_dir * hid_dim)
                                                )
            # _zk.shape = (_n_sample * n_batch, latent_dim)
            # _sum_log_jacobian.shape = (_n_sample * n_batch, )
            _zk = _zk.view(_n_sample, B, self.latent_dim)  # shape = (_n_sample, n_batch, latent_dim)
            _sum_log_jacobian = _sum_log_jacobian.view(_n_sample, B)  # shape = (_n_sample, n_batch)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            _init_states = gpu_wrapper(torch.zeros([self.enc_layers, _n_sample * B, self.n_dir * self.hid_dim])).float()  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)

            # ----- Importance sampling for NLL -----
            _logits = self.Decoder(init_states=_init_states,  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)
                                   latent_vector=_zk.view(_n_sample * B, self.latent_dim),  # shape = (_n_sample * n_batch, out_dim)
                                   helper=go.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 15)
                                   test_lm=True)  # shape = (_n_sample * n_batch, 16, V)
            _xent = self.criterionSeq(_logits,  # shape = (_n_sample * n_batch, 16, V)
                                      eos.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 16)
                                      keep_batch=True).view(_n_sample, B)  # shape = (_n_sample, n_batch)

            _log_pzk = self.PriorGaussian.log_prob(_zk).sum(2)  # shape = (_n_sample, n_batch)
            _log_pxzk = - _xent  # shape = (_n_sample, n_batch)
            _log_qzkx = Q0.log_prob(_z0).sum(2) - _sum_log_jacobian  # shape = (_n_sample, n_batch)

            samplify['xent'].append(_xent)  # shape = (_n_sample, n_batch)
            samplify['log_pzk'].append(_log_pzk)  # shape = (_n_sample, n_batch)
            samplify['log_pxzk'].append(_log_pxzk)  # shape = (_n_sample, n_batch)
            samplify['log_qzkx'].append(_log_qzkx)  # shape = (_n_sample, n_batch)
            samplify['zk'].append(_zk)  # shape = (_n_sample, n_batch, out_dim)

        for key in samplify.keys():
            samplify[key] = torch.cat(samplify[key], dim=0)  # shape = (n_sample, ?)

        ll = log_sum_exp(samplify['log_pzk'] + samplify['log_pxzk'] - samplify['log_qzkx'], dim=0) - np.log(n_sample)  # shape = (n_batch, )
        nll = - ll  # shape = (n_batch, )

        # ----- Importance sampling for KL -----
        # kl = kl_with_isogaussian(gaussian_dist)  # shape = (n_batch, )
        kl = (samplify['log_qzkx'] - samplify['log_pzk']).mean(0)  # shape = (n_batch, )

        return samplify['xent'].mean(0), nll, kl, samplify['zk'].transpose(0, 1)

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
        Q0, z0 = self.PosteriorGaussian(last_states)
        # z0.shape = (n_batch, latent_dim)

        # ----- Sampling -----
        z0 = Q0.rsample(torch.Size([n_sample]))  # shape = (n_sample, n_batch, out_dim)
        assert tuple(z0.shape) == (n_sample, B, self.latent_dim)

        # ----- Flows -----
        zk, _ = self.Flows(z0=z0.contiguous().view(n_sample * B, self.latent_dim),
                            # shape = (n_sample * n_batch, out_dim)
                            cond=last_states.unsqueeze(0).expand(n_sample, -1, -1).contiguous().view(n_sample * B, -1)
                            # shape = (n_sample * n_batch, layers * n_dir * hid_dim)
                            )
        # _zk.shape = (n_sample * n_batch, latent_dim)
        zk = zk.view(n_sample, B, self.latent_dim)  # shape = (n_sample, n_batch, latent_dim)

        samples = zk  # shape = (n_sample, n_batch, latent_dim)
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
            last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

            # ----- Posterior Network -----
            Q0, z0 = self.PosteriorGaussian(last_states)
            # z0.shape = (n_batch, latent_dim)

            # ----- Flows -----
            zk, sum_log_jacobian = self.Flows(z0=z0, cond=last_states)
            # zk.shape = (n_batch, latent_dim)
            # sum_log_jacobian.shape = (n_batch, )

            # ----- Bag-of-Words logits -----
            BoW_logits = self.BoW(zk)  # shape = (n_bathc, voc_size)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                latent_vector=zk,
                                helper=go), Q0, z0, zk, sum_log_jacobian, BoW_logits


class Gaussian(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Gaussian, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.toMiu = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.BatchNorm1d(self.out_dim))
        self.toSigma = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.BatchNorm1d(self.out_dim))

    def reinit(self):
        self.toSigma._modules['1'].bias.data.zero_()  # TODO
        pass

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
