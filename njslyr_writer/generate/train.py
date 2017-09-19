#!/opt/conda/bin/python3

import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import pickle
import json
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser
import string
import numpy as np
from chainer import optimizers, Variable
from njslyr_writer.generate.rnn import RNN

preprocessed_dir = Path('/preprocessed')
models_dir = Path('/models')

def jsonload(path):
    with open(path, 'r') as f:
        return json.load(f)

def readbatch(scenario_num=3, batch_size=30, fill=-1):
    scenarios = np.random.choice(list(preprocessed_dir.glob('tokens/*.json')), scenario_num)
    tweets = np.random.choice(sum(map(jsonload, scenarios), []), batch_size)
    data = np.ones([batch_size, max(map(len, tweets))], dtype=np.int32) * fill 
    for i, tweet in enumerate(tweets):
        data[i,:len(tweet)] = tweet
    return data.astype(np.int32)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=20000)
    parser.add_argument('-n', '--n_unit', type=int, default=40)
    parser.add_argument('-z', '--optimizer', type=str, default="MomentumSGD")
    parser.add_argument('-o', '--output', type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    with open(preprocessed_dir / 'dictionaries' / 'word2idx.json', 'r') as f:
        word2idx = json.load(f)
    with open(preprocessed_dir / 'dictionaries' / 'bag_of_words.json', 'r') as f:
        bow = {int(k):x for k, x in json.load(f).items()}

    lstm = RNN(args.n_unit, bow)
    optimizer = getattr(optimizers, args.optimizer)()
    optimizer.setup(lstm)

    it = args.iter
    for i in range(it):
        data = readbatch()
        seq = Variable(data.T)
        lstm.reset_state()
        optimizer.update(lstm.compute_loss, seq)

    (models_dir / 'generators').mkdir(exist_ok=True)

    dump_name = args.output if args.output is not None else ''.join(np.random.choice(list(string.ascii_letters + string.digits), 8)) + '.pkl'
    with open (models_dir / 'generators' / dump_name, 'wb') as f:
        pickle.dump(lstm, f)

if __name__ == '__main__':
    main()
