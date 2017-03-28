import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import sys
import pickle
import json
from argparse import ArgumentParser
from os import listdir
from collections import Counter
import numpy as np
from rnn import RNN

def pick_head():
    head = None
    while head is None:
        try:
            with open('/home/ninja/translated/' + np.random.choice(listdir('/home/ninja/translated')), 'r') as f:
                s = json.load(f)
                head = np.random.choice(s)[0]
        except ValueError:
            continue
    return head

def write(model, head=None):
    with open('/home/ninja/translated/dicts/words.json', 'r') as f:
        words = json.load(f)
    word2ind = {w: i for i, w in enumerate(words)}
    if head is None:
        head = pick_head()
    else:
        head = word2ind[head]
    with open(model, 'rb') as f:
        lstm = pickle.load(f)
    generated = "".join([words[i] for i in lstm.sample(head, word2ind['EOS'])])
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
