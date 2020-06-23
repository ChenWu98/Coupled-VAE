import os
import shutil
from Experiment import Experiment
from config import Config

config = Config()


def test_rec():

    # Building the wrapper
    wrapper = Experiment(test=True)

    if config.train_mode == 'gen':
        wrapper.restore_model(['AutoEncoder'])
    else:
        raise ValueError()

    wrapper.sample_from_prior()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    test_rec()
