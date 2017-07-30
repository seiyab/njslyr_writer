import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import pickle
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from rnn import RNN

preprocessed_dir = Path('/preprocessed')
models_dir = Path('/models')

def pick_head():
    with open(np.random.choice(list(preprocessed_dir.glob('tokens/*.json'))), 'r') as f:
        tweets = json.load(f)
        head = np.random.choice(tweets)[0]
    return head

def write(generator, head=None):
    with open(preprocessed_dir / 'dictionaries' / 'word2idx.json', 'r') as f:
        word2idx = json.load(f)
    words = [-1] * len(word2idx)
    for w, i in word2idx.items():
        words[i] = w
    if head is None:
        head = pick_head()
    else:
        head = word2idx[head]
    with open(generator, 'rb') as f:
        lstm = pickle.load(f)
    generated = "".join([words[i] for i in lstm.sample(head, word2idx['EOS'])])
    return generated.rstrip('EOS')

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('--head', type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    generated = write(args.model, head=args.head)
    print(generated)
    print("{} characters".format(str(len(generated))))

if __name__ == "__main__":
    main()
