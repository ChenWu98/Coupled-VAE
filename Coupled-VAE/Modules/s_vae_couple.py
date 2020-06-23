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
from Modules.Losses.SeqLoss import SeqLoss
from Modules.vae_nf_geo import _n_sample
from Modules.subModules.ive import ive
from torch.distributions.kl import register_kl
from Modules.vae_couple import Decoder
from utils.utils import gpu_wrapper
from config import Config

config = Config()


class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    # @property
    # def dim(self):
    #     return self._dim

    # @property
    # def device(self):
    #     return self._device

    # @device.setter
    # def device(self, val):
    #     self._device = val if isinstance(val, torch.device) else torch.device(val)

    # def __init__(self, dim, validate_args=None, device="cpu"):
    def __init__(self, dim, validate_args=None):
        """
        Hypersphere in the dim-dimensional space. (In the original implementation, dim is the dimension of
        the hypersphere, which might be misleading. Modified by Chen Wu)
        """
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        # self.device = device

    def sample(self, shape=torch.Size()):
        output = gpu_wrapper(torch.distributions.Normal(0, 1).sample((shape if isinstance(shape, torch.Size) else torch.Size([shape])) + torch.Size([self._dim])))

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return - gpu_wrapper(torch.ones(x.shape[:-1])) * self.__log_surface_area()

    def __log_surface_area(self):
        return np.log(2) + (self._dim / 2) * np.log(np.pi) - torch.lgamma(gpu_wrapper(torch.Tensor([self._dim / 2])))


class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.__m = loc.shape[-1]
        self.__e1 = gpu_wrapper(torch.Tensor([1.] + [0] * (loc.shape[-1] - 1)))

        super(VonMisesFisher, self).__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)

        v = (gpu_wrapper(torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape))).transpose(0, -1)[1:]).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = gpu_wrapper(torch.distributions.Uniform(0, 1).sample(shape))
        self.__w = 1 + torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(torch.max(gpu_wrapper(torch.tensor([0.])), self.scale - 10), gpu_wrapper(torch.tensor([1.])))
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * np.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w

    def __while_loop(self, b, a, d, shape):

        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b), torch.zeros_like(b), (torch.ones_like(b) == 1)

        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            e_ = gpu_wrapper(torch.distributions.Beta((self.__m - 1) / 2, (self.__m - 1) / 2).sample(shape[:-1]).reshape(shape))
            u = gpu_wrapper(torch.distributions.Uniform(0, 1).sample(shape))

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * t.log() - t + d) > torch.log(u)
            reject = 1 - accept

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e, w

    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        output = - self.scale * ive(self.__m / 2, self.scale) / ive((self.__m / 2) - 1, self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale) - (self.__m / 2) * np.log(2 * np.pi) - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return - vmf.entropy() + hyu.entropy()


class S_VAE_COUPLE(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(S_VAE_COUPLE, self).__init__()
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
        self.PriorUniform = HypersphericalUniform(dim=self.latent_dim)
        self.PosteriorVMF = VonMisesFisherModule(in_dim=self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim)
        self.PosteriorVMFCouple = VonMisesFisherModule(in_dim=self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim, no_instance=True)
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
        self.DecoderCouple = Decoder(voc_size=self.voc_size,
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

        self.toInit = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.emb_dim, self.emb_dim))
        self.toInitCouple = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.emb_dim, self.emb_dim))

    def estimate_mi(self, go, sent_len, bare, n_sample):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        vmf_dist, _ = self.PosteriorVMF(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        mi, sampled_latents = self.importance_sampling_mi(vmf_dist=vmf_dist, n_sample=n_sample)  # shape = (n_batch, )

        return mi, sampled_latents

    def importance_sampling_mi(self, vmf_dist, n_sample):
        assert n_sample % _n_sample == 0

        B = vmf_dist.mean.shape[0]

        samplify = {
            'log_qz': [],
            'log_qzx': [],
            'z': []
        }
        for sample_id in range(n_sample // _n_sample):
            # ----- Sampling -----
            _z = vmf_dist.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, latent_dim)
            assert tuple(_z.shape) == (_n_sample, B, self.latent_dim)

            _log_qzx = vmf_dist.log_prob(_z)  # shape = (_n_sample, n_batch)
            _log_qz = vmf_dist.log_prob(_z.unsqueeze(2).expand(-1, -1, B, -1))  # shape = (_n_sample, n_batch, n_batch)
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
        vmf_dist, _ = self.PosteriorVMF(last_states)
        # _.shape = (n_batch, latent_dim)

        # ----- Importance sampling estimation -----
        xent, nll, kl, sampled_latents = self.importance_sampling(vmf_dist=vmf_dist, go=go, eos=eos, n_sample=n_sample)
        # xent.shape = (n_batch, )
        # nll.shape = (n_batch, )
        # kl.shape = (n_batch, )
        # sampled_latents.shape = (n_batch, n_sample, latent_dim)

        return xent, nll, kl, sampled_latents

    def importance_sampling(self, vmf_dist, go, eos, n_sample):
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
            _z = vmf_dist.rsample(torch.Size([_n_sample]))  # shape = (_n_sample, n_batch, latent_dim)
            assert tuple(_z.shape) == (_n_sample, B, self.latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            _init_states = gpu_wrapper(torch.zeros([self.enc_layers, _n_sample * B, self.n_dir * self.hid_dim])).float()  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)

            _init_input = self.toInit(_z)  # shape = (_n_sample, n_batch, emb_dim)

            # ----- Importance sampling for NLL -----
            _logits = self.Decoder(init_states=_init_states,  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)
                                   init_input=_init_input.view(_n_sample * B, self.emb_dim),  # shape = (_n_sample * n_batch, out_dim)
                                   helper=go.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 15)
                                   test_lm=True)  # shape = (_n_sample * n_batch, 16, V)
            _xent = self.criterionSeq(_logits,  # shape = (_n_sample * n_batch, 16, V)
                                      eos.unsqueeze(0).expand(_n_sample, -1, -1).contiguous().view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, 16)
                                      keep_batch=True).view(_n_sample, B)  # shape = (_n_sample, n_batch)

            _log_pz = self.PriorUniform.log_prob(_z)  # shape = (_n_sample, n_batch)
            _log_pxz = - _xent  # shape = (_n_sample, n_batch)
            _log_qzx = vmf_dist.log_prob(_z)  # shape = (_n_sample, n_batch)

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
        kl = (samplify['log_qzx'] - samplify['log_pz']).mean(0)  # shape = (n_batch, )

        return samplify['xent'].mean(0), nll, kl, samplify['z'].transpose(0, 1)

    def generate_uniform(self, B):
        return self.PriorUniform.sample(torch.Size([B]))  # shape = (n_batch, emb_dim)

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
        vmf_dist, _ = self.PosteriorVMF(last_states)

        samples = vmf_dist.sample(torch.Size([n_sample]))  # shape = (n_sample, n_batch, latent_dim)
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

        init_input = self.toInit(latents)  # shape = (n_batch, emb_dim)

        return self.Decoder(init_states=init_states,
                            init_input=init_input,
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
            latent_vector = self.generate_uniform(B)  # shape = (n_batch, latent_dim)

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
            vmf_dist, latent_vector = self.PosteriorVMF(last_states)
            # latent_vector.shape = (n_batch, latent_dim)
            _, latent_vector_couple = self.PosteriorVMFCouple(last_states)
            # latent_vector_couple.shape = (n_batch, latent_dim)

            prior_unif = HypersphericalUniform(dim=self.latent_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            init_input = self.toInit(latent_vector)  # shape = (n_batch, emb_dim)
            # init_input_couple = self.toInitCouple(vmf_dist_couple.loc)  # shape = (n_batch, emb_dim)
            init_input_couple = self.toInitCouple(latent_vector_couple)  # shape = (n_batch, emb_dim)

            logits = self.Decoder(init_states=init_states,
                                  init_input=init_input,
                                  helper=go)
            logits_couple = self.DecoderCouple(init_states=init_states,
                                                   init_input=init_input_couple,
                                                   helper=go)

            return logits, vmf_dist, prior_unif, logits_couple, init_input, init_input_couple


class VonMisesFisherModule(nn.Module):

    def __init__(self, in_dim, out_dim, no_instance=False):
        super(VonMisesFisherModule, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.no_instance = no_instance

        self.toMiu = nn.Sequential(nn.Linear(self.in_dim, self.out_dim))
        self.toKappa = nn.Sequential(nn.Linear(self.in_dim, 1))

    def forward(self, input):
        """

        :param input: shape = (n_batch, in_dim)
        :return: vmf_dist is a Distribution object and latent_vector.shape = (n_batch, out_dim)
        """

        miu = self.toMiu(input)  # shape = (n_batch, out_dim)
        miu = miu / miu.norm(dim=-1, keepdim=True)  # shape = (n_batch, out_dim)
        kappa = F.softplus(self.toKappa(input)) + 1  # shape = (n_batch, 1)
        if self.no_instance:
            return None, miu
        else:
            vmf_dist = VonMisesFisher(miu, kappa)
            latent_vector = vmf_dist.rsample(torch.Size([1])).squeeze(0)  # shape = (1, n_batch, out_dim) -> (n_batch, out_dim)

            return vmf_dist, latent_vector
