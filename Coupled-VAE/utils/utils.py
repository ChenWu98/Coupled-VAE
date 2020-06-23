import torch
import numpy as np
import h5py
import torch.nn.functional as F

from config import Config
config = Config()


def gpu_wrapper(item):
    if config.gpu:
        # print(item)
        return item.cuda()
    else:
        return item


def sort_resort(length):
    s_idx = [ix for ix, l in sorted(enumerate(length.data.cpu()), key=lambda x: x[1], reverse=True)]
    res_idx = [a for a, b in sorted(enumerate(s_idx), key=lambda x: x[1])]
    return s_idx, res_idx


def strip_eos(sents):
    if isinstance(sents[0], list):
        return [sent[:max(sent.index('<eos>'), 1)] if '<eos>' in sent else sent
                for sent in sents]
    else:
        sent = sents
        return sent[:max(sent.index('<eos>'), 1)] if '<eos>' in sent else sent


def strip_pad(sents):
    if isinstance(sents[0], list):
        return [sent[:max(sent.index('<pad>'), 1)] if '<pad>' in sent else sent
                for sent in sents]
    else:
        sent = sents
        return sent[:max(sent.index('<pad>'), 1)] if '<pad>' in sent else sent


def gumbel_softmax(logits, gamma, eps=1e-20):
    """ logits.shape = (..., voc_size) """
    U = torch.zeros_like(logits).uniform_()
    G = -torch.log(-torch.log(U + eps) + eps)
    return F.softmax((logits + G) / gamma, dim=-1)


def pretty_string(flt):
    ret = '%.4f' % flt  # TODO
    if flt >= 0:
        ret = "+" + ret
    return ret


def sample_2d(probs, temperature):
    """probs.shape = (n_batch, n_choices)"""
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), config.temp_att))  # shape = (n_batch, 20)
    else:
        temp = probs
    sample_idx = torch.multinomial(temp, 1)  # shape = (n_batch, 1)
    sample_probs = probs.gather(1, sample_idx)  # shape = (n_batch, 1)
    sample_idx = sample_idx.squeeze(1)  # shape = (n_batch, )
    sample_probs = sample_probs.squeeze(1)  # shape = (n_batch, )
    return sample_idx, sample_probs


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def create_null_mask(lengths, max_len):
    """

    :param lengths: shape = (n_batch, )
    :param max_len: int.
    :return: shape = (n_batch, max_len)
    """
    return gpu_wrapper(1 - torch.arange(0, max_len)
                       .type_as(lengths)
                       .repeat(lengths.shape[0], 1)
                       .lt(lengths.unsqueeze(1)))


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp


def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


def compute_rouge_l(output, reference, mode='f', max_len=None):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    if max_len is None:
        max_len = len(output)
    max_len = min(max_len, len(output))
    output = output[:max_len]
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = 1.0 * lcs / len(output)
        recall = 1.0 * lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score