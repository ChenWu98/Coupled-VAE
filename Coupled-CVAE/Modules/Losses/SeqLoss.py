import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper


class SeqLoss(nn.Module):
    def __init__(self, voc_size, pad, end, unk):
        super(SeqLoss, self).__init__()
        self.voc_size = voc_size
        self.pad = pad
        self.word_weight = gpu_wrapper(torch.ones(voc_size))
        self.word_weight[pad] = 0.

    def forward(self, logits, gts, keep_batch=False):
        """
        :param logits: (?, T, V)
        :param gts: (?, T)
        :param keep_batch: bool.
        :return: Scalar or (?).
        """
        if logits.shape[0] == 0:
            assert gts.shape[0] == 0
            return gpu_wrapper(torch.FloatTensor([0])).squeeze(0)

        assert logits.shape[:-1] == gts.shape
        if not keep_batch:
            xent = F.cross_entropy(input=logits.contiguous().view(-1, self.voc_size),
                                   target=gts.view(-1),
                                   weight=self.word_weight)
            return xent
        else:
            T = logits.shape[-2]
            stuct_shape = list(logits.shape[:-2])
            xent = F.cross_entropy(input=logits.contiguous().view(-1, self.voc_size),
                                   target=gts.view(-1),
                                   weight=self.word_weight,
                                   reduction='none')
            xent = xent.view(stuct_shape + [T])  # shape = (?, T)
            xent = xent.sum(-1)  # shape = (?)
            return xent
