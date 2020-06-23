import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from utils.utils import sort_resort

from config import Config

config = Config()


class Encoder(nn.Module):

    def __init__(self, emb_dim, hid_dim, n_layer, dropout, bi, embedding):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.bi = bi
        self.n_dir = 2 if self.bi else 1

        self.drop = nn.Dropout(self.dropout)
        self.Embedding = embedding
        self.use_lstm = config.use_lstm
        if self.use_lstm:
            self.LSTM = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=self.hid_dim,
                                num_layers=self.n_layer,
                                dropout=self.dropout,
                                bidirectional=self.bi)
        else:
            self.GRU = nn.GRU(input_size=self.emb_dim,
                              hidden_size=self.hid_dim,
                              num_layers=self.n_layer,
                              dropout=self.dropout,
                              bidirectional=self.bi)

    def forward(self, bare, length):
        """

        :param bare: shape = (non_pad, 50)
        :param length: shape = (non_pad, )
        :return:
        """
        B, T = bare.shape
        # Sort.
        s_idx, res_idx = sort_resort(length)
        bare = bare[s_idx, :]  # shape = (non_pad, 50)
        length = length[s_idx]  # shape = (non_pad, )

        # Encode.
        embeded = self.Embedding(bare).transpose(0, 1)  # shape = (50, non_pad, emb_dim)
        packed = pack(self.drop(embeded), length)
        if self.use_lstm:
            outputs, (last_states, _) = self.LSTM(packed)
        else:
            outputs, last_states = self.GRU(packed)
        outputs = unpack(outputs, total_length=T)[0]
        outputs = outputs.transpose(0, 1)
        # outputs.shape = (non_pad, 50, n_dir * hid_dim)
        # last_states.shape = (layers * n_dir, non_pad, hid_dim)

        # Resort.
        outputs = outputs[res_idx, :, :]  # shape = (non_pad, 50, n_dir * hid_dim)
        last_states = last_states[:, res_idx, :]  # shape = (layers * n_dir, non_pad, hid_dim)

        return outputs, last_states
