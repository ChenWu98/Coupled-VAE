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


class SEQ2SEQ(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(SEQ2SEQ, self).__init__()
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

        self.criterionSeq = SeqLoss(voc_size=self.voc_size, pad=pad_id, end=None, unk=None)

    def test_lm(self, post_bare, post_len, resp_go, resp_len, resp_bare, resp_eos, n_sample):
        B = post_bare.shape[0]

        # ----- Post Encoding -----
        post_outputs, post_last_states = self.PostEncoder(post_bare, post_len)
        # post_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # post_last_states.shape = (layers * n_dir, n_batch, hid_dim)
        post_last_states = post_last_states.transpose(0, 1).contiguous().view(B, -1)  # shape = (n_batch, layers * n_dir * hid_dim)
        post_repr = self.PostRepr(post_last_states)  # shape = (n_batch, emb_dim)

        # ----- Initial Decoding States -----
        assert self.enc_bi
        init_states = gpu_wrapper(torch.zeros(
            [self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        logits = self.Decoder(init_states=init_states,
                              post_repr=post_repr,
                              latent_vector=None,
                              helper=resp_go,
                              test_lm=True)

        # ----- Importance sampling estimation -----
        xent = self.criterionSeq(logits,
                                 resp_eos,
                                 keep_batch=True)  # shape = (n_batch, )
        nll = xent

        return xent, nll, torch.zeros_like(xent)

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

        # ----- Initial Decoding States -----
        assert self.enc_bi
        init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        preds = self.Decoder(init_states=init_states,
                             post_repr=post_repr,
                             latent_vector=None,
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

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                post_repr=post_repr,
                                latent_vector=None,
                                helper=resp_go)
