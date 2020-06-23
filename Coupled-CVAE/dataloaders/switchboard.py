from torch.utils import data
import torch
import os
from collections import defaultdict
import numpy as np
from utils.vocab import Vocabulary, build_vocab
import random
from nltk import word_tokenize
import json
from config import Config


config = Config()
np.random.seed(0)


class SwitchBoard(data.Dataset):
    """The SwitchBoard dataset."""

    def __init__(self, mode):
        self.mode = mode
        assert self.mode in ['train', 'valid', 'test']
        self.root = os.path.join('data', 'switchboard')
        voc_f = os.path.join(self.root, 'switchboard.vocab')
        self.max_len = config.max_len

        self.posts = []
        self.responses = []
        with open(os.path.join(self.root, '{}.txt'.format(self.mode))) as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                post, response = line.strip().split('\t')
                self.posts.append(post.split())
                self.responses.append(response.split())

        print('SwitchBoard data successfully read.')

        # Build vocabulary.
        if self.mode == 'train':
            print('----- Building vocab -----')
            build_vocab(self.posts + self.responses, voc_f, min_occur=5)  # TODO

        # Load vocabulary.
        print('----- Loading vocab -----')
        self.vocab = Vocabulary(voc_f)
        print('vocabulary size:', self.vocab.size)
        self.pad = self.vocab.word2id['<pad>']
        self.go = self.vocab.word2id['<go>']
        self.eos = self.vocab.word2id['<eos>']
        self.unk = self.vocab.word2id['<unk>']

    def get_references(self):
        raise NotImplementedError()

    def get_sentences(self):
        posts = []
        responses = []
        for index in range(len(self)):
            post = self.posts[index]
            posts.append(post)
            response = self.responses[index]
            responses.append(response)
        return posts, responses

    def process_raw_sent(self, sent, max_len):
        l = len(sent)
        sent_id = [self.vocab.word2id[w] if w in self.vocab.word2id else self.unk for w in sent]
        if l > max_len:
            sent_id = sent_id[:max_len]
            l = max_len
        padding = [self.pad] * (max_len - l)
        bare = torch.LongTensor(sent_id + padding)  # shape = (20, )
        go = torch.LongTensor([self.go] + sent_id + padding)  # shape = (21, )
        eos = torch.LongTensor(sent_id + [self.eos] + padding)  # shape = (21, )
        return bare, go, eos, torch.LongTensor([l]).squeeze()

    def __getitem__(self, index):
        raw_post = self.posts[index]
        raw_resp = self.responses[index]
        post_bare, post_go, post_eos, post_len = self.process_raw_sent(raw_post, self.max_len)
        resp_bare, resp_go, resp_eos, resp_len = self.process_raw_sent(raw_resp, self.max_len)

        return post_bare, post_len, resp_bare, resp_go, resp_eos, resp_len

    def __len__(self):
        assert len(self.posts) == len(self.responses)
        return len(self.posts)

