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
from Modules.decoder import Decoder
from Modules.Losses.SeqLoss import SeqLoss


class DAE(nn.Module):

    def __init__(self, hid_dim, latent_dim, enc_layers, dec_layers, dropout, enc_bi, dec_max_len, beam_size, WEAtt_type, encoder_emb, decoder_emb, pad_id):
        super(DAE, self).__init__()
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
        self.toLatent = nn.Linear(self.hid_dim * self.n_dir * self.enc_layers, self.latent_dim)
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

    def visualize(self, go, sent_len, bare):
        B = bare.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B,
                                                                    -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        samples = self.toLatent(last_states)  # shape = (n_batch, latent_dim)

        return samples

    def test_lm(self, go, sent_len, bare, eos, n_sample):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        latent_vector = self.toLatent(last_states.transpose(0, 1).contiguous().view(B, -1))  # shape = (n_batch, latent_dim)

        # ----- Initial Decoding States -----
        assert self.enc_bi
        init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        logits = self.Decoder(init_states=init_states,
                              latent_vector=latent_vector,
                              helper=go,
                              test_lm=True)  # shape = (n_batch, 16, V)
        xent = self.criterionSeq(logits, eos, keep_batch=True)  # shape = (n_batch, )
        kl = torch.zeros_like(xent) + float('inf')  # shape = (n_batch, )

        nll = xent + kl  # shape = (n_batch, )

        return xent, nll, kl, latent_vector

    def generate_gaussian(self, B):
        return self.PriorGaussian.sample(torch.Size([B]))  # shape = (n_batch, emb_dim)

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
            latent_vector = self.toLatent(last_states.transpose(0, 1).contiguous().view(B, -1))  # shape = (n_batch, emb_dim)

            # ----- Initial Decoding States -----
            assert self.enc_bi
            init_states = gpu_wrapper(torch.zeros([self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

            return self.Decoder(init_states=init_states,
                                latent_vector=latent_vector,
                                helper=go), latent_vector

    def saliency(self, go, sent_len=None, bare=None):
        B = go.shape[0]

        # ----- Encoding -----
        outputs, last_states = self.Encoder(bare, sent_len)
        # ext_outputs.shape = (n_batch, 15, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, n_batch, hid_dim)
        last_states = last_states.transpose(0, 1).contiguous().view(B,
                                                                    -1)  # shape = (n_batch, layers * n_dir * hid_dim)

        # ----- Posterior Network -----
        latent_vector = self.toLatent(last_states)
        # latent_vector.shape = (n_batch, latent_dim)

        # ----- Initial Decoding States -----
        assert self.enc_bi
        init_states = gpu_wrapper(torch.zeros(
            [self.enc_layers, B, self.n_dir * self.hid_dim])).float()  # shape = (layers, n_batch, n_dir * hid_dim)

        logits = self.Decoder(init_states=init_states,
                              latent_vector=latent_vector,
                              helper=go)

        return logits, self.Decoder.toInit(latent_vector), last_states
