import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import pickle
import json
from os import listdir, chdir
import sys
from collections import Counter
from argparse import ArgumentParser
import numpy as np
from chainer import optimizers, Variable
from rnn import RNN

def readbatch(batch_size=10, fill=-1):
    tweets = None
    while tweets is None:
        try:
            with open('translated/' + np.random.choice(listdir("translated")), 'r') as f:
                tweets = np.random.choice(json.load(f), size=batch_size)
        except:
            continue
    data = np.ones([batch_size, max(map(len, tweets))], dtype=np.int32) * fill 
    for i, tweet in enumerate(tweets):
        data[i,:len(tweet)] = tweet
    return data.astype(np.int32)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('-i', '--iter', type=int, default=20000)
    parser.add_argument('-n', '--n_unit', type=int, default=40)
    parser.add_argument('-z', '--optimizer', type=str, default="Adam")
    return parser.parse_args()

def main():
    chdir('/home/ninja')
    args = get_args()
    savefile = args.model
    if not savefile.endswith('.pkl'):
        raise ValueError('filename must end with .pkl')
    with open('translated/dicts/words.json', 'r') as f:
        words = json.load(f)
    with open('translated/dicts/bag_of_ids.json', 'r') as f:
        bow = {int(k):x for k, x in json.load(f).items()}
    word2ind = {w: i for i, w in enumerate(words)}

    lstm = RNN(args.n_unit, bow)
    optimizer = getattr(optimizers, args.optimizer)()
    optimizer.setup(lstm)

    it = args.iter
    for i in range(it):
        data = readbatch()
        seq = Variable(data.T)
        lstm.reset_state()
        optimizer.update(lstm.compute_loss, seq)

    with open (args.model, 'wb') as f:
        pickle.dump(lstm, f)

main()
