import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils.utils import gpu_wrapper


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sample_logprobs, reward, mask=None):
        """

        :param sample_logprobs: shape = (n_batch, *)
        :param mask: shape = (n_batch, *) or None
        :param reward: shape = (n_batch, *)
        :return:
        """
        sample_logprobs = sample_logprobs.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        if mask is not None:
            mask = mask.float().contiguous().view(-1)
            output = - sample_logprobs * reward * mask
            output = torch.sum(output) / torch.sum(mask)
        else:
            output = - sample_logprobs * reward
            output = output.mean()
        return output
