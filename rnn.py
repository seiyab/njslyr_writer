import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import pickle
import json
from os import listdir
import sys
from collections import Counter
import numpy as np
import chainer
from chainer import optimizers, Chain, Variable
import chainer.functions as F
import chainer.links as L

class RNN(Chain):
    def __init__(self, n_units, bow):
        self.words = list(bow.keys())
        self.n_vocab = len(self.words)
        super(RNN, self).__init__(
            embed=L.EmbedID(self.n_vocab, n_units),
            l1 = L.LSTM(n_units, n_units),
            l2 = L.NegativeSampling(n_units, [bow[i] for i in range(len(bow))], 10)
        )
        self.word2ind = {word: i for i, word in enumerate(self.words)}

    def __call__(self, x, t):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        loss = self.l2(h1, t)
        return loss

    def reset_state(self):
        self.l1.reset_state()

    def compute_loss(self, x_list):
        loss = 0
        self.reset_state()
        for cur_word, next_word in zip(x_list, x_list[1:]):
            loss += self(cur_word, next_word)
        return loss

    def sample(self, head, EOS, max_length=100, number_of_candidates=30, softmax_temperature=4, eos_c=0.2):
        self.reset_state()
        ret = [head]
        for i in range(max_length):
            h0 = self.embed(Variable(np.array([ret[-1]], dtype=np.int32)))
            h1 = self.l1(h0)
            v = F.log_softmax(F.matmul(h1, self.l2.W, transb=True)).data[0] * softmax_temperature
            v[EOS] += i*eos_c
            cands = np.argsort(v)[-number_of_candidates:]
            ps = np.exp(v[cands] - v[cands].max())
            ps /= ps.sum()
            ret.append(np.random.choice(cands, p=ps))
            if ret[-1]==EOS:
                break
        return ret
