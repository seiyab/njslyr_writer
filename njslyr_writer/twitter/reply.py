#!/opt/conda/bin/python3

from pathlib import Path
import json
import numpy as np
import MeCab
from njslyr_writer.twitter import twpy
from njslyr_writer.generate import write as w

models_dir = Path('/models')
dictionaries_dir = Path('/preprocessed') / 'dictionaries'
twitter_config_dir = Path('/config/twitter')

m = MeCab.Tagger("-Ochasen")

def reply_all():
    with open(dictionaries_dir / 'word2idx.json', 'r') as f:
        word2ind = json.load(f)
    with open(twitter_config_dir / 'account.json', 'r') as f:
        replies = get_replies(json.load(f)['id'], 30, get_lastid()+1)
    for text, id_, user in reversed(replies):
        head = m.parse(text).split('\n')[0].split('\t')[0]
        twpy.api.update_status(generate_reply(user, head), in_reply_to_status_id=id_)
        update_lastid(id_)

def get_replies(target, count, since_id):
    at_tweets = twpy.api.search('@{}'.format(target), count=count, since_id=since_id)
    replies = [t for t in at_tweets if t.in_reply_to_screen_name==target]
    texts = (r.text.replace('@{}'.format(target), '').strip() for r in replies)
    ids = (r.id for r in replies)
    screen_names = (r.user.screen_name for r in replies)
    return list(zip(texts, ids, screen_names))

def generate_reply(user, head):
    generator = np.random.choice(list(models_dir.glob('generators/*.pkl')))
    if head=='EOS':
        text = w.write(generator)
    else:
        try:
            text = w.write(generator, head)
        except:
            text = '◆失敗な◆単語「{}」がワカリマセンドスエ'.format(head)
    tweet = "@{} {}".format(user, text)
    return tweet[:140]

def update_lastid(i):
    with open(twitter_config_dir / 'tweet_id.json', 'w') as f:
        json.dump(i, f)

def get_lastid():
    with open(twitter_config_dir / 'tweet_id.json', 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    reply_all()
