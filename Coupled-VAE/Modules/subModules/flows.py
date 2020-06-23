import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config

config = Config()


class PlanarNormalizingFlow(nn.Module):

    def __init__(self, cond_dim, latent_dim):
        super(PlanarNormalizingFlow, self).__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.cond_params = config.cond_params  # TODO

        if self.cond_params:
            self.mapu = nn.Sequential(nn.Linear(self.cond_dim, self.latent_dim),
                                      nn.BatchNorm1d(self.latent_dim))
            self.mapw = nn.Sequential(nn.Linear(self.cond_dim, self.latent_dim),
                                      nn.BatchNorm1d(self.latent_dim))
            self.mapb = nn.Sequential(nn.Linear(self.cond_dim, 1),
                                      nn.BatchNorm1d(1))
        else:
            self.u = nn.Parameter(torch.Tensor(1, self.latent_dim))  # shape = (n_batch, latent_dim)
            self.w = nn.Parameter(torch.Tensor(1, self.latent_dim))  # shape = (n_batch, latent_dim)
            self.b = nn.Parameter(torch.Tensor(1, 1))  # shape = (n_batch, 1)

            self.u.data.uniform_(-0.01, 0.01)
            self.w.data.uniform_(-0.01, 0.01)
            self.b.data.uniform_(-0.01, 0.01)

    def forward(self, z, cond):
        """

        :param z: shape = (n_batch, latent_dim)
        :param cond: shape = (n_batch, cond_dim)
        :return:
        """
        B = z.shape[0]

        if self.cond_params:
            u = self.mapu(cond)  # shape = (n_batch, latent_dim)
            w = self.mapw(cond)  # shape = (n_batch, latent_dim)
            b = self.mapb(cond)  # shape = (n_batch, 1)
        else:
            u = self.u  # shape = (1, latent_dim)
            w = self.w  # shape = (1, latent_dim)
            b = self.b  # shape = (1, 1)

        # Create uhat such that it is parallel to w
        uw = torch.sum(u * w, dim=1, keepdim=True)  # shape = (n_batch, 1)
        muw = -1 + F.softplus(uw)  # shape = (n_batch, 1)
        uhat = u + (muw - uw) * w / torch.sum(w ** 2, dim=1, keepdim=True)  # shape = (n_batch, latent_dim)

        # Equation 21 - Transform z
        zwb = torch.sum(z * w, dim=1, keepdim=True) + b  # shape = (n_batch, 1)
        assert zwb.shape[0] == B and zwb.shape[1] == 1

        f_z = z + (uhat * torch.tanh(zwb))  # shape = (n_batch, latent_dim) + (n_batch, latent_dim) * (n_batch, 1) -> (n_batch, latent_dim)
        psi = (1 - torch.tanh(zwb) ** 2) * w  # shape = (n_batch, 1) * (n_batch, latent_dim) -> (n_batch, latent_dim)
        psi_u = torch.sum(psi * uhat, dim=1)  # shape = (n_batch, )
        assert psi_u.shape[0] == B and len(psi_u.shape) == 1

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        return f_z, logdet_jacobian


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """
    def __init__(self, cond_dim, latent_dim, n_flows, flow_type):
        super(NormalizingFlows, self).__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.n_flows = n_flows
        self.flow_type = flow_type

        if flow_type == 'planar':
            self.flows = nn.ModuleList([PlanarNormalizingFlow(cond_dim=self.cond_dim, latent_dim=self.latent_dim) for _ in range(self.n_flows)])
        elif flow_type == 'radial':
            raise NotImplementedError()
        else:
            raise ValueError()

    def forward(self, z0, cond):
        """

        :param z0: shape = (n_batch, latent_dim)
        :param cond: shape = (n_batch, cond_dim)
        :return: zk.shape = (n_batch, latent_dim); sum_log_jacobian.shape = (n_batch, )
        """
        sum_log_jacobian = 0
        z = z0  # shape = (n_batch, latent_dim)
        for flow in self.flows:
            z, log_jacobian = flow(z=z, cond=cond)
            sum_log_jacobian = sum_log_jacobian + log_jacobian  # shape = (n_batch, )

        zk = z  # shape = (n_batch, latent_dim)

        return zk, sum_log_jacobian
