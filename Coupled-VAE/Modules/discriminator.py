import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gpu_wrapper


class Discriminator(nn.Module):

    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.dim = dim

        # self.model = nn.Sequential(
        #     nn.Linear(self.dim, 1)
        # )
        self.model = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1)
        )

    def forward(self, x):
        """

        :param x: shape = (n_batch, max_len, dim_h)
        :return: shape = (n_batch, max_len, 1)
        """
        return self.model(x)

