import os
from utils.multi_bleu import calc_bleu_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk import bigrams
from tqdm import tqdm
import torch
from utils.vocab_eval import Vocabulary

smooth = SmoothingFunction()
EPSILON = 1e-10


class Evaluator(object):
    def __init__(self, dataset, vocab):
        pass

    def eval_BLEU(self, predss, refss):
        assert len(predss) == len(refss)
        p_BLEU, r_BLEU, f_BLEU = [], [], []
        for preds, refs in zip(predss, refss):
            _p_BLEU, _r_BLEU = self._eval_BLEU(preds, refs)
            _f_BLEU = harmonic(_p_BLEU, _r_BLEU)
            p_BLEU.append(_p_BLEU)
            r_BLEU.append(_r_BLEU)
            f_BLEU.append(_f_BLEU)
        return mean(p_BLEU), mean(r_BLEU), mean(f_BLEU)

    def eval_DIST(self, preds):
        uni_grams = []
        bi_grams = []
        for pred in preds:
            uni_grams.extend(pred)
            bi_grams.extend(bigrams(pred))
        dist_1 = len(set(uni_grams)) / len(uni_grams)
        dist_2 = len(set(bi_grams)) / len(bi_grams)

        return dist_1, dist_2

    @staticmethod
    def _eval_BLEU(preds, refs):
        n_pred_sample = len(preds)
        n_ref_sample = len(refs)

        log_dir = 'outputs/temp_results'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        pxr_scores = [[0 for ref_id in range(n_ref_sample)] for pred_id in range(n_pred_sample)]  # n_pred * n_ref
        rxp_scores = [[0 for ref_id in range(n_ref_sample)] for pred_id in range(n_pred_sample)]  # n_pred * n_ref
        for pred_id in range(n_pred_sample):
            for ref_id in range(n_ref_sample):
                # multi_BLEU = calc_bleu_score(
                #     [' '.join(preds[pred_id])],
                #     [[' '.join(refs[ref_id])]],
                #     log_dir=log_dir,
                #     multi_ref=True)
                nltk_BLEU = sentence_bleu(
                    references=[refs[ref_id]],
                    hypothesis=preds[pred_id],
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth.method3  # TODO
                )
                pxr_scores[pred_id][ref_id] = nltk_BLEU
                rxp_scores[ref_id][pred_id] = nltk_BLEU

        _p_BLEU = mean([max(scores) for scores in pxr_scores])
        _r_BLEU = mean([max(scores) for scores in rxp_scores])
        return _p_BLEU, _r_BLEU


def harmonic(a, b):
    return 2 / (1/(a+EPSILON) + 1/(b+EPSILON))


def mean(seq):
    return sum(seq) / len(seq)