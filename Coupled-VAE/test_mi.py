import os
import shutil
from Experiment import Experiment
from config import Config

config = Config()


def test_mi():

    # Building the wrapper
    wrapper = Experiment(test=True)

    if config.train_mode == 'gen':
        wrapper.restore_model(['AutoEncoder'])
    else:
        raise ValueError()

    wrapper.estimate_mi()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    test_mi()
