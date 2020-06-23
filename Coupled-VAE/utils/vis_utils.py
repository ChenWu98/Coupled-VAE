import os
import numpy as np
import math
import torch
import networkx as nx

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from config import Config

config = Config()
plt.switch_backend('agg')


def tensor2im(input_image, imtype=np.uint8):
    """input_image.shape = (?, H, W)"""
    image_numpy = input_image.data.cpu().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


class Visualizer(object):
    def __init__(self, ncols):
        self.ncols = ncols  # Number of rows.
        self.vis = []

    def add_batch(self, image_list):
        assert len(image_list) == self.ncols, '{} != {}'.format(len(image_list), self.ncols)
        self.vis.append(torch.stack(image_list, dim=1))  # shape = (n_batch, self.ncols, 3, H, W)

    def to_image(self):
        vis = torch.cat(self.vis, dim=0)  # shape = (num_generated*n_batch, self.ncols, 3, H, W)
        H = vis.shape[3]
        W = vis.shape[4]
        return tensor2im(make_grid(vis.view(-1, 3, H, W), nrow=self.ncols))
        # nrow in make_grid means the number of images in each row!


class MyMesh(object):

    def __init__(self,
                 height,
                 width,
                 n_rows,
                 n_cols,
                 ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig = plt.figure(figsize=(height, width))
        self.axes = self.fig.subplots(n_rows, n_cols, squeeze=False)

    def get_axis(self, row, col):
        assert (0 <= row < self.n_rows) and (0 <= col < self.n_cols)
        return self.axes[row, col]

    def get_fig(self):
        return self.fig
