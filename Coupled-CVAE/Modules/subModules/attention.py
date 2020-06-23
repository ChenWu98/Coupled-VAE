import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
config = Config()


class AttUnitTanh(nn.Module):

    def __init__(self, query_dim, key_dim, atten_dim):
        super(AttUnitTanh, self).__init__()

        self.atten_dim = atten_dim
        self.W = nn.Linear(query_dim, atten_dim)  # Applied on query.
        self.U = nn.Linear(key_dim, atten_dim)  # Applied on keys.
        self.v = nn.Linear(atten_dim, 1)

    def forward(self, queries, keys, null_mask):
        """
        Compute the attention of each query on each key.

        :param queries: shape = (seq_len, n_batch, query_dim)
        :param keys: shape = (keys_per_batch, key_dim) or (n_batch, keys_per_batch, key_dim)
        :param null_mask: shape = (keys_per_batch, ) or (n_batch, keys_per_batch)
        :return attened_keys.shape = (n_batch, seq_len, key_dim); att_weight.shape = (seq_len, n_batch, keys_per_batch)
        """

        if len(keys.shape) == 2:
            # All batches share the same keys.
            keys = keys.unsqueeze(0).expand(queries.shape[1], -1, -1)  # shape = (n_batch, keys_per_batch, key_dim)
        else:
            assert queries.shape[1] == keys.shape[0]
        if null_mask is not None and len(null_mask.shape) == 1:
            # All batches share the same null_mask.
            null_mask = null_mask.unsqueeze(0).expand(queries.shape[1], -1)  # shape = (n_batch, keys_per_batch)

        t_key = self.U(keys).unsqueeze(0)  # shape =      (1,       n_batch, keys_per_batch, atten_dim)
        t_query = self.W(queries).unsqueeze(2)  # shape = (seq_len, n_batch, 1,              atten_dim)
        alpha = self.v(torch.tanh(t_query + t_key)).squeeze(3)  # shape = (seq_len, n_batch, keys_per_batch)
        if null_mask is not None:
            null_mask = null_mask.unsqueeze(0)  # shape = (1, n_batch, keys_per_batch)
            alpha.masked_fill_(null_mask, -float('inf'))
        att_weight = F.softmax(alpha, dim=2)  # shape = (seq_len, n_batch, keys_per_batch)
        attened_keys = torch.bmm(att_weight.transpose(0, 1), keys).transpose(0, 1)  # shape = (seq_len, n_batch, key_dim)

        return attened_keys, alpha


class AttUnitBiLi(nn.Module):

    def __init__(self, query_dim, key_dim):
        super(AttUnitBiLi, self).__init__()

        self.linear_in = nn.Linear(query_dim, key_dim, bias=False)

    def forward(self, queries, keys, null_mask):
        """
        Compute the attention of each query on each key.

        :param queries: shape = (seq_len, n_batch, query_dim)
        :param keys: shape = (keys_per_batch, key_dim) or (n_batch, keys_per_batch, key_dim)
        :param null_mask: shape = (keys_per_batch, ) or (n_batch, keys_per_batch)
        :return attened_keys.shape = (n_batch, seq_len, key_dim); att_weight.shape = (seq_len, n_batch, keys_per_batch)
        """
        if len(keys.shape) == 2:
            # All batches share the same keys.
            keys = keys.unsqueeze(0).expand(queries.shape[1], -1, -1)  # shape = (n_batch, keys_per_batch, key_dim)
        else:
            assert queries.shape[1] == keys.shape[0]
        if null_mask is not None and len(null_mask.shape) == 1:
            # All batches share the same null_mask.
            null_mask = null_mask.unsqueeze(0).expand(queries.shape[1], -1)  # shape = (n_batch, keys_per_batch)

        t_query = self.linear_in(queries).transpose(0, 1)  # shape = (n_batch, seq_len, key_dim)
        alpha = torch.bmm(t_query, keys.transpose(1, 2)).transpose(0, 1)  # shape = (seq_len, n_batch, keys_per_batch)
        if null_mask is not None:
            null_mask = null_mask.unsqueeze(0)  # shape = (1, n_batch, keys_per_batch)
            alpha.masked_fill_(null_mask, -float('inf'))
        att_weight = F.softmax(alpha, dim=2)  # shape = (seq_len, n_batch, keys_per_batch)
        attened_keys = torch.bmm(att_weight.transpose(0, 1), keys).transpose(0, 1)  # shape = (seq_len, n_batch, key_dim)

        return attened_keys, alpha


class AttUnitDot(nn.Module):

    def __init__(self):
        super(AttUnitDot, self).__init__()

    def forward(self, queries, keys, null_mask):
        """
        Compute the attention of each query on each key.

        :param queries: shape = (seq_len, n_batch, query_dim)
        :param keys: shape = (keys_per_batch, key_dim) or (n_batch, keys_per_batch, key_dim)
        :param null_mask: shape = (keys_per_batch, ) or (n_batch, keys_per_batch)
        :return attened_keys.shape = (n_batch, seq_len, key_dim); att_weight.shape = (seq_len, n_batch, keys_per_batch)
        """
        assert queries.shape[2] == keys.shape[-1]
        if len(keys.shape) == 2:
            # All batches share the same keys.
            keys = keys.unsqueeze(0).expand(queries.shape[1], -1, -1)  # shape = (n_batch, keys_per_batch, key_dim)
        else:
            assert queries.shape[1] == keys.shape[0]
        if null_mask is not None and len(null_mask.shape) == 1:
            # All batches share the same null_mask.
            null_mask = null_mask.unsqueeze(0).expand(queries.shape[1], -1)  # shape = (n_batch, keys_per_batch)

        alpha = torch.bmm(queries.transpose(0, 1), keys.transpose(1, 2)).transpose(0, 1)  # (seq_len, n_batch , keys_per_batch)
        if null_mask is not None:
            null_mask = null_mask.unsqueeze(0)  # shape = (1, n_batch, keys_per_batch)
            alpha.masked_fill_(null_mask, -float('inf'))
        att_weight = F.softmax(alpha, dim=2)  # shape = (seq_len, n_batch, keys_per_batch)
        attened_keys = torch.bmm(att_weight.transpose(0, 1), keys).transpose(0, 1)  # shape = (seq_len, n_batch, key_dim)

        return attened_keys, alpha