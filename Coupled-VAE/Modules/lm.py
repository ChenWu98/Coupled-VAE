import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper, gumbel_softmax
from utils.utils import strip_eos
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from Modules.encoder import Encoder
from Modules.subModules.attention import AttUnitBiLi, AttUnitDot, AttUnitTanh
from Modules.Losses.SeqLoss import SeqLoss
from config import Config

config = Config()


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
        self.toInit = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.emb_dim, self.emb_dim))  # TODO
        self.use_lstm = config.use_lstm
        if self.use_lstm:
            self.LSTM = nn.LSTM(input_size=self.emb_dim * 2,
                                hidden_size=self.hid_dim,
                                num_layers=self.n_layers,
                                dropout=self.dropout,
                                bidirectional=False)
        else:
            self.GRU = nn.GRU(input_size=self.emb_dim,
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

    def forward(self, init_states, latent_vector, helper, test_lm=False):
        """

        :param init_states: shape = (n_layer, n_batch, hid_dim), which should be zero if not seq2seq
        :param latent_vector: shape = (n_batch, latent_dim) as the initial input of the GRU
        :param helper: shape = (n_batch, 16)
        :param test_lm: True if we are testing the language modeling performance.
        :return: shape = (n_batch, 16, voc_size) or [?]
        """
        B, T = helper.shape

        # Flatten.
        h_0 = None  # shape = (n_layer, n_batch, hid_dim)  # TODO: only used in this project!

        if self.training or test_lm:
            # ----- Teacher forcing -----
            helper_emb = self.Embedding(helper).transpose(0, 1)  # shape = (51, n_batch, emb_dim)
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
            init_state = BeamState(h_0, inp, [[] for _ in range(BN)], [0] * BN)
            beam = [init_state]

            for t in range(self.max_len):
                exp = [[] for _ in range(BN)]
                for state in beam:
                    if self.use_lstm:
                        outputs, h = self.LSTM(self.drop(state.inp), state.h)
                    else:
                        outputs, h = self.GRU(self.drop(state.inp), state.h)
                    # output.shape = (1, non_pad, hid_dim); h.shape = (n_layer, non_pad, hid_dim)
                    final = F.tanh(self.W_c(self.drop(outputs)))  # shape = (1, non_pad, emb_dim)
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
                    embs = self.drop(self.Embedding(indices))  # shape = (1, non_pad, beam_size, emb_dim)
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
            h = h_0
            sents = []
            for t in range(self.max_len):
                if self.use_lstm:
                    outputs, h = self.LSTM(self.drop(inp), h)
                else:
                    outputs, h = self.GRU(self.drop(inp), h)
                # output.shape = (1, BN, hid_dim); h.shape = (n_layer, BN, hid_dim)
                final = F.tanh(self.W_c(self.drop(outputs)))  # shape = (1, BN, emb_dim)
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

                inp = self.drop(self.Embedding(index))  # shape = (1, BN, emb_dim)
            sents = [[sents[l][:, b] for l in range(self.max_len)] for b in range(BN)]

            return sents


class LM(nn.Module):

    def __init__(self, hid_dim, latent_dim, dec_layers, dropout, dec_max_len, beam_size, WEAtt_type, decoder_emb, pad_id):
        super(LM, self).__init__()
        self.voc_size = decoder_emb.num_embeddings
        self.emb_dim = decoder_emb.embedding_dim
        self.hid_dim = hid_dim
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.n_dir = 2  # Kept for dimension consistency.
        self.dec_max_len = dec_max_len
        self.beam_size = beam_size
        self.WEAtt_type = WEAtt_type
        self.latent_dim = latent_dim  # Kept for nothing.

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

    def test_lm(self, go, sent_len, bare, eos, n_sample):
        B = go.shape[0]

        latent_vectors = torch.zeros(B, self.latent_dim)  # shape = (n_batch, latent_dim)

        # ----- Initial Decoding States -----
        init_states = gpu_wrapper(torch.zeros([self.dec_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        logits = self.Decoder(init_states=init_states,
                              latent_vector=None,
                              helper=go,
                              test_lm=True)  # shape = (n_batch, 16, V)
        xent = self.criterionSeq(logits, eos, keep_batch=True)  # shape = (n_batch, )
        kl = torch.zeros_like(xent)  # shape = (n_batch, )

        nll = xent + kl  # shape = (n_batch, )

        return xent, nll, kl, latent_vectors

    def forward(self, go, sent_len=None, bare=None):
        """

        :param go: shape = (n_batch, 16)
        :param sent_len: shape = (n_batch, ) or None
        :param bare: shape = (n_batch, 15) or None
        :return:
        """
        B = go.shape[0]

        if not self.training:
            raise NotImplementedError()
        else:

            # ----- Initial Decoding States -----
            init_states = gpu_wrapper(torch.zeros([self.dec_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                latent_vector=None,
                                helper=go)
