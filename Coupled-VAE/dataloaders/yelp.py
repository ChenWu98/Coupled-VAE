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


class Yelp_Interp(data.Dataset):
    """The Yelp dataset for Interpolation."""
    def __init__(self):
        self.root = os.path.join('data', 'yelp')
        voc_f = os.path.join(self.root, 'yelp.vocab')
        self.max_len = config.max_len

        self.sentence_pairs = []
        sentences = []
        with open(os.path.join(self.root, 'interp_pairs.txt')) as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                sent = line.strip().split()
                sentences.append(sent)
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                self.sentence_pairs.append((sentences[i], sentences[j]))
        print('Yelp Interpolation data successfully read.')

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

    def process_raw_sent(self, sent, max_len):
        l = len(sent)
        # sent_id = [self.vocab.word2id[w] if w in self.vocab.word2id else self.unk for w in sent]
        sent_id = [self.vocab.word2id[w] for w in sent]  # TODO: this line is correct only if min_occur == 1.
        if l > max_len:
            sent_id = sent_id[:max_len]
            l = max_len
        padding = [self.pad] * (max_len - l)
        bare = torch.LongTensor(sent_id + padding)  # shape = (20, )
        go = torch.LongTensor([self.go] + sent_id + padding)  # shape = (21, )
        eos = torch.LongTensor(sent_id + [self.eos] + padding)  # shape = (21, )
        return bare, go, eos, torch.LongTensor([l]).squeeze()

    def __getitem__(self, index):
        sentA_raw, sentB_raw = self.sentence_pairs[index]
        sent_bareA, sent_goA, sent_eosA, sent_lenA = self.process_raw_sent(sentA_raw, self.max_len)
        sent_bareB, sent_goB, sent_eosB, sent_lenB = self.process_raw_sent(sentB_raw, self.max_len)

        return sent_bareA, sent_goA, sent_eosA, sent_lenA, sent_bareB, sent_goB, sent_eosB, sent_lenB

    def __len__(self):
        return len(self.sentence_pairs)


class Yelp(data.Dataset):
    """The Yelp dataset."""

    def __init__(self, mode):
        self.mode = mode
        assert self.mode in ['train', 'valid', 'test']
        self.root = os.path.join('data', 'yelp')
        voc_f = os.path.join(self.root, 'yelp.vocab')
        self.max_len = config.max_len

        self.sentences = []
        with open(os.path.join(self.root, '{}.txt'.format(self.mode))) as f:
            for line in f.readlines():
                if len(line.strip()) in [0, 1]:
                    continue
                words = line.strip().split()
                assert words[0] in [str(dig) for dig in range(5)], '{} does not start with the rating'.format(words)
                self.sentences.append(words[1:])

        print('Yelp data successfully read.')

        # Build vocabulary.
        if self.mode == 'train':
            print('----- Building vocab -----')
            build_vocab(self.sentences, voc_f, min_occur=1)  # TODO

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
        sents = []
        for index in range(len(self)):
            sent = self.sentences[index]
            sents.append(sent)
        return sents

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
        raw_sent = self.sentences[index]
        sent_bare, sent_go, sent_eos, sent_len = self.process_raw_sent(raw_sent, self.max_len)

        return sent_bare, sent_go, sent_eos, sent_len

    def __len__(self):
        return len(self.sentences)


class YelpWithLabel(data.Dataset):
    """The Yelp dataset with labels."""

    def __init__(self, mode):
        self.mode = mode
        assert self.mode in ['train', 'valid', 'test']
        self.root = os.path.join('data', 'yelp')
        voc_f = os.path.join(self.root, 'yelp.vocab')
        self.max_len = config.max_len

        self.labels = []
        self.sentences = []
        with open(os.path.join(self.root, '{}.txt'.format(self.mode))) as f:
            for line in f.readlines():
                if len(line.strip()) in [0, 1]:
                    continue
                words = line.strip().split()
                assert words[0] in [str(dig) for dig in range(5)], '{} does not start with the rating'.format(words)
                self.sentences.append(words[1:])
                self.labels.append(int(words[0]))

        print('Yelp data successfully read.')

        # Build vocabulary.
        if self.mode == 'train':
            print('----- Building vocab -----')
            build_vocab(self.sentences, voc_f, min_occur=1)  # TODO

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
        sents = []
        for index in range(len(self)):
            sent = self.sentences[index]
            sents.append(sent)
        return sents

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
        raw_sent = self.sentences[index]
        label = self.labels[index]
        sent_bare, sent_go, sent_eos, sent_len = self.process_raw_sent(raw_sent, self.max_len)

        return sent_bare, sent_go, sent_eos, sent_len, torch.LongTensor([label]).squeeze()

    def __len__(self):
        return len(self.sentences)