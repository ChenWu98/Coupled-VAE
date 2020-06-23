import numpy as np
from numpy import linalg as LA
import torch
import pickle
import random
from collections import Counter
from utils.word_vectors import get_embedding
from config import Config
import os
config = Config()


class Vocabulary(object):
    def __init__(self, vocab_file, for_eval=False):
        with open(vocab_file, 'rb') as f:
            self.size, self.word2id, self.id2word = pickle.load(f)

        print('Loading word vectors FOR EVALUATION ONLY.')
        self.embedding = get_embedding(names=self.id2word, wv_dim=config.emb_dim, for_eval=for_eval)


def build_vocab(data, path, min_occur=5):
    if os.path.exists(path):
        return
    id2word = ['<pad>', '<go>', '<eos>', '<unk>', '<mask>']
    word2id = {tok: ix for ix, tok in enumerate(id2word)}

    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)
    with open(path, 'wb') as f:
        pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)
