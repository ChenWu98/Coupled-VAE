import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

sns.set(color_codes=True)
sns.set(style="white")


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

    def save_fig(self, save_name):
        self.fig.savefig(save_name)


def main():

    for dataset in ['pwkp', 'ewsew']:
        mesh = MyMesh(6.4 * 1, 4.8 * 1, 1, 1)
        data = pd.read_excel('/Users/wuchen/Desktop/ACL-results/change-of-m-{}.xlsx'.format(dataset))
        ax = mesh.get_axis(0, 0)
        sns.lineplot(data=data, x='m', y='value', hue='Metric', ax=ax, markers=True, style="Metric")
        if dataset == 'pwkp':
            ax.set_ylim([0, 60])  # TODO
            ax.set_xlim([0.2, 0.5])
            ax.set_xticks([0.2, 0.3, 0.4, 0.5])
        else:
            ax.set_ylim([-15, 90])  # TODO
            ax.set_xlim([0.2, 0.5])
            ax.set_xticks([0.2, 0.3, 0.4, 0.5])
        ax.set_ylabel(r'value')
        ax.set_xlabel(r'$m$')
        handles, l = ax.get_legend_handles_labels()
        ax.legend(handles, l, loc='lower right', framealpha=None)  # TODO
        # axY.tick_params(labelsize=25)
        mesh.save_fig('/Users/wuchen/Desktop/change-of-m-{}.pdf'.format(dataset))

    for dataset in ['pwkp', 'ewsew']:
        mesh = MyMesh(6.4 * 1, 4.8 * 1, 1, 1)
        data = pd.read_excel('/Users/wuchen/Desktop/ACL-results/change-of-bias-{}.xlsx'.format(dataset))
        ax = mesh.get_axis(0, 0)
        sns.lineplot(data=data, x='bias', y='value', hue='Metric', ax=ax, markers=True, style="Metric")
        if dataset == 'pwkp':
            ax.set_ylim([-3, 50])  # TODO
            ax.set_xlim([1.0, 2.5])
            ax.set_xticks([1.0, 1.5, 2.0, 2.5])
        else:
            ax.set_ylim([-25, 100])  # TODO
            ax.set_xlim([0.2, 0.8])
            ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_ylabel(r'value')
        ax.set_xlabel(r'$\gamma_{s}$')
        handles, l = ax.get_legend_handles_labels()
        ax.legend(handles, l, loc='lower right', framealpha=None)  # TODO
        # axY.tick_params(labelsize=25)
        mesh.save_fig('/Users/wuchen/Desktop/change-of-bias-{}.pdf'.format(dataset))



if __name__ == '__main__':
    main()
