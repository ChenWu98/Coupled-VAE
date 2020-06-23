"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
convert Switchboard (Tiancheng Zhao's version, see following url) to txt files
https://github.com/snakeztc/NeuralDialog-CVAE/tree/master/data
"""

import pickle, json, sys, io, nltk, os


def cvae_tokenize(s):
	# follow Tiancheng's code
	return ' '.join(nltk.WordPunctTokenizer().tokenize(s.lower()))


def p2txt(path):
	data = pickle.load(open(path, 'rb'))
	for k in data:
		lines = []
		for i, d in enumerate(data[k]):
			if i%100 == 0:
				print('[%s]%i/%i'%(k, i, len(data[k])))
			txts = [cvae_tokenize(txt) for spk, txt, feat in d['utts']]
			for t in range(1, len(txts) - 1):
				# src = ' EOS '.join(txts[:t])
				src = txts[t-1]
				tgt = txts[t]
				lines.append(src + '\t' + tgt)
		with open(os.path.join('data/switchboard/', k + '.txt'), 'w') as f:
			f.write('\n'.join(lines))


if __name__ == '__main__':
	path_orig = 'data/switchboard/full_swda_clean_42da_sentiment_dialog_corpus.p'
	p2txt(path_orig)
