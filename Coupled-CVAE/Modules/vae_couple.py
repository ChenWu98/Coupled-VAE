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

from torch.autograd import Variable
from utils.utils import gpu_wrapper
from Modules.subModules.attention import AttUnitBiLi, AttUnitDot, AttUnitTanh
from copy import deepcopy
from config import Config

config = Config()
_n_sample = 2


class BeamState(object):
    def __init__(self, h, inp, sent, nll):
        """
        Wrapper of a real beam or an intermediate beam.
        :param h:      real: shape = (n_layers, n_batch, dim_h)        intermediate: shape = (n_layers, dim_h)
        :param inp:    real: shape = (1, n_batch, emb_dim)             intermediate: shape = (1, emb_dim)
        :param sent:   real: [[shape = (1, ) x cur_len] x n_batch]     intermediate: [shape = (1, ) x cur_len]
        :param nll:    real: [scalar x n_batch]                        intermediate: scalar
        """
        self.h, self.inp, self.sent, self.nll = h, inp, sent, nll


class Decoder(nn.Module):

    def __init__(self, voc_size, latent_dim, emb_dim, hid_dim, n_layer, dropout, max_len, beam_size, WEAtt_type, embedding):
        super(Decoder, self).__init__()
        self.voc_size = voc_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layer
        self.dropout = dropout
        self.max_len = max_len
        self.beam_size = beam_size
        self.WEAtt_type = WEAtt_type
        self.test_non_unk = [False, True][1]  # TODO

        self.drop = nn.Dropout(dropout)
        self.Embedding = embedding

        self.use_lstm = config.use_lstm
        if self.use_lstm:
            self.LSTM = nn.LSTM(input_size=self.emb_dim * 2,
                                hidden_size=self.hid_dim,
                                num_layers=self.n_layers,
                                dropout=self.dropout,
                                bidirectional=False)
        else:
            self.GRU = nn.GRU(input_size=self.emb_dim * 2,
                              hidden_size=self.hid_dim,
                              num_layers=self.n_layers,
                              dropout=self.dropout,
                              bidirectional=False)
        self.W_c = nn.Linear(self.hid_dim, self.emb_dim)
        if self.WEAtt_type == 'tanh':
            self.WEAttention = AttUnitTanh(query_dim=self.emb_dim,
                                           key_dim=self.emb_dim,
                                           atten_dim=self.emb_dim)
        elif self.WEAtt_type == 'dot':
            self.WEAttention = AttUnitDot()
        elif self.WEAtt_type == 'bilinear':
            self.WEAttention = AttUnitBiLi(query_dim=self.emb_dim,
                                           key_dim=self.emb_dim)
        elif self.WEAtt_type == 'none':
            self.ToVocab = nn.Linear(self.emb_dim, self.voc_size)
        else:
            raise ValueError()

    def forward(self, init_states, init_input, helper, test_lm=False):
        """

        :param init_states: shape = (n_layer, n_batch, hid_dim), which should be zero if not seq2seq
        :param init_input: shape = (n_batch, emb_dim) as the initial input of the GRU
        :param helper: shape = (n_batch, 16)
        :param test_lm: True if we are testing the language modeling performance.
        :return: shape = (n_batch, 16, voc_size) or [?]
        """

        init_input = init_input.unsqueeze(0)  # shape = (1, n_batch, emb_dim)

        # Flatten.
        h_0 = None  # shape = (n_layer, n_batch, hid_dim)  # TODO: only used in this project!
        if self.use_lstm:
            _, h_0 = self.LSTM(torch.cat([init_input, init_input], dim=2), h_0)  # h_0.shape = (n_layer, n_batch, hid_dim)
        else:
            _, h_0 = self.GRU(torch.cat([init_input, init_input], dim=2), h_0)  # h_0.shape = (n_layer, n_batch, hid_dim)

        if self.training or test_lm:
            # ----- Teacher forcing -----
            helper_emb = self.Embedding(helper).transpose(0, 1)  # shape = (51, n_batch, emb_dim)
            cond = init_input.expand(helper.shape[1], -1, -1)  # shape = (51, n_batch, emb_dim)
            helper_emb = torch.cat([helper_emb, cond], dim=2)  # shape = (51, n_batch, emb_dim * 2)  TODO: added.

            if self.use_lstm:
                outputs, _ = self.LSTM(self.drop(helper_emb), h_0)  # shape = (51, n_batch, hid_dim)
            else:
                outputs, _ = self.GRU(self.drop(helper_emb), h_0)  # shape = (51, n_batch, hid_dim)
            final = F.tanh(self.W_c(self.drop(outputs)))  # shape = (51, n_batch, emb_dim)
            if self.WEAtt_type != 'none':
                _, WE_alpha = self.WEAttention(final, self.Embedding.weight, None)  # shape = (51, n_batch, V)
            else:
                WE_alpha = self.ToVocab(final)  # shape = (16, n_batch, V)
            logits = WE_alpha.transpose(0, 1)  # shape = (non_pad, 51, V)
            return logits
        elif self.beam_size > 1:
            # ----- Inference beam search -----
            BN = helper.shape[0]
            inp = self.Embedding(helper[:, 0:1]).transpose(0, 1)  # shape = (1, non_pad, emb_dim)
            cond = init_input  # shape = (1, n_batch, emb_dim)
            inp = torch.cat([inp, cond], dim=2)  # shape = (1, non_pad, emb_dim * 2)
            init_state = BeamState(h_0, inp, [[] for _ in range(BN)], [0] * BN)
            beam = [init_state]

            for t in range(self.max_len):
                exp = [[] for _ in range(BN)]
                for state in beam:
                    if self.use_lstm:
                        outputs, h = self.LSTM(state.inp, state.h)
                    else:
                        outputs, h = self.GRU(state.inp, state.h)
                    # output.shape = (1, non_pad, hid_dim); h.shape = (n_layer, non_pad, hid_dim)
                    final = F.tanh(self.W_c(outputs))  # shape = (1, non_pad, emb_dim)
                    if self.WEAtt_type != 'none':
                        _, WE_alpha = self.WEAttention(final, self.Embedding.weight, None)  # shape = (1, non_pad, V)
                    else:
                        WE_alpha = self.ToVocab(final)  # shape = (1, non_pad, V)
                    logits = WE_alpha.transpose(0, 1)  # shape = (non_pad, 1, V)
                    # TODO: note that we mask <unk> here. Assert that <unk> is of index 3.
                    if self.test_non_unk:
                        logits[:, :, 3] = - float('inf')

                    logits = logits.transpose(0, 1)  # shape = (1, non_pad, V)
                    log_lh = F.log_softmax(logits, dim=2)  # shape = (1, non_pad, V)
                    log_lh, indices = torch.topk(log_lh, self.beam_size, dim=2)
                    # log_lh.shape = (1, non_pad, beam_size); indices.shape = (1, non_pad, beam_size)
                    embs = self.Embedding(indices)  # shape = (1, non_pad, beam_size, emb_dim)
                    cond = init_input.unsqueeze(2).expand(-1, -1, self.beam_size, -1).contiguous()  # shape = (1, n_batch, beam_size, emb_dim)
                    embs = torch.cat([embs, cond], dim=3)  # shape = (1, non_pad, beam_size, emb_dim * 2)
                    for i in range(BN):
                        for l in range(self.beam_size):
                            exp[i].append(BeamState(h[:, i, :],  # shape = (n_layers, dim_h)
                                                    embs[:, i, l],  # shape = (1, emb_dim)
                                                    state.sent[i] + [indices[:, i, l]],  # A list of (1, )s.
                                                    state.nll[i] - log_lh[0, i, l]))  # Scalar tensor.

                beam = [BeamState(torch.zeros_like(h_0).copy_(h_0),
                                  torch.zeros_like(inp).copy_(inp),
                                  [[] for _ in range(BN)],
                                  [0] * BN)
                        for _ in range(self.beam_size)]
                for i in range(BN):
                    a = sorted(exp[i], key=lambda x: x.nll)
                    for k in range(self.beam_size):
                        beam[k].h[:, i, :] = a[k].h
                        beam[k].inp[:, i, :] = a[k].inp
                        beam[k].sent[i] = a[k].sent
                        beam[k].nll[i] = a[k].nll

            return beam[0].sent
        else:  # Beam size == 1
            # ----- Inference greedy search -----
            BN = helper.shape[0]
            inp = self.Embedding(helper[:, 0:1]).transpose(0, 1)  # shape = (1, BN, emb_dim)
            cond = init_input  # shape = (1, n_batch, emb_dim)
            inp = torch.cat([inp, cond], dim=2)  # shape = (1, BN, emb_dim * 2)
            h = h_0
            sents = []
            for t in range(self.max_len):
                if self.use_lstm:
                    outputs, h = self.LSTM(inp, h)
                else:
                    outputs, h = self.GRU(inp, h)
                # output.shape = (1, BN, hid_dim); h.shape = (n_layer, BN, hid_dim)
                final = F.tanh(self.W_c(outputs))  # shape = (1, BN, emb_dim)
                if self.WEAtt_type != 'none':
                    _, WE_alpha = self.WEAttention(final, self.Embedding.weight, None)  # shape = (1, BN, V)
                else:
                    WE_alpha = self.ToVocab(final)  # shape = (1, n_batch * n_sum, V)
                logits = WE_alpha.transpose(0, 1)  # shape = (non_pad, 1, V)
                # TODO: note that we mask <unk> here. Assert that <unk> is of index 3.
                if self.test_non_unk:
                    logits[:, :, 3] = - float('inf')

                logits = logits.transpose(0, 1)  # shape = (1, BN, V)
                index = torch.argmax(logits, dim=2)  # shape = (1, BN)
                sents.append(index)

                inp = self.Embedding(index)  # shape = (1, BN, emb_dim)
                inp = torch.cat([inp, cond], dim=2)  # shape = (1, BN, emb_dim * 2)
            sents = [[sents[l][:, b] for l in range(self.max_len)] for b in range(BN)]

            return sents


class VAE_COUPLE(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(VAE_COUPLE, self).__init__()
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
        self.PosteriorGaussianCouple = Gaussian(in_dim=2 * self.hid_dim * self.n_dir * self.enc_layers, out_dim=self.latent_dim)
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

        self.BoW = nn.Linear(self.latent_dim, self.voc_size)
        self.PostRepr = nn.Linear(self.hid_dim * self.n_dir * self.enc_layers, self.emb_dim)

        self.criterionSeq = SeqLoss(voc_size=self.voc_size, pad=pad_id, end=None, unk=None)

        self.toInit = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.emb_dim, self.emb_dim))
        self.toInitCouple = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.emb_dim, self.emb_dim))

    def test_lm(self, post_bare, post_len, resp_go, resp_len, resp_bare, resp_eos, n_sample):
        B = post_bare.shape[0]

        # ----- Post Encoding -----
        post_outputs, post_last_states = self.PostEncoder(post_bare, post_len)
        # post_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # post_last_states.shape = (layers * n_dir, n_batch, hid_dim)
        post_last_states = post_last_states.transpose(0, 1).contiguous().view(B,
                                                                              -1)  # shape = (n_batch, layers * n_dir * hid_dim)
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

            init_input = self.toInit(_z) + post_repr.unsqueeze(0).expand(_n_sample, -1, -1).contiguous()  # shape = (_n_sample, n_batch, emb_dim)

            # ----- Importance sampling for NLL -----
            _logits = self.Decoder(init_states=_init_states,  # shape = (layers, _n_sample * n_batch, n_dir * hid_dim)
                                   init_input=init_input.view(_n_sample * B, -1),  # shape = (_n_sample * n_batch, emb_dim)
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

        init_input = self.toInit(prior_latent) + post_repr  # shape = (n_batch, emb_dim)

        preds = self.Decoder(init_states=init_states,
                             init_input=init_input,
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
            posterior_dist_couple, _ = self.PosteriorGaussianCouple(torch.cat([resp_last_states, post_last_states], dim=1))

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            init_input = self.toInit(posterior_latent) + post_repr  # shape = (n_batch, emb_dim)
            init_input_couple = self.toInitCouple(posterior_dist_couple.mean) + post_repr  # shape = (n_batch, emb_dim)

            logits = self.Decoder(init_states=init_states,
                                  init_input=init_input,
                                  helper=resp_go)
            logits_couple = self.DecoderCouple(init_states=init_states,
                                                   init_input=init_input_couple,
                                                   helper=resp_go)

            return logits, prior_dist, posterior_dist, logits_couple, init_input, init_input_couple


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
