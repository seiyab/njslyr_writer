import MeCab
from pathlib import Path
import json

m = MeCab.Tagger("-Ochasen")

data_dir = Path('/data')
preprocessed_dir = Path('/preprocessed')

def main():
    preprocessed_scenarios_dir = preprocessed_dir / 'tokens'
    preprocessed_scenarios_dir.mkdir(exist_ok=True)
    dictionary_dir = preprocessed_dir / 'dictionaries'
    dictionary_dir.mkdir(exist_ok=True)

    word2idx = {}
    for chapter in filter(lambda x: x.is_dir(), data_dir.iterdir()):
        for scenario in filter(lambda x: x.suffix=='.json', chapter.iterdir()):
            with open(scenario, 'r') as f:
                sentences = json.load(f)
                preprocessed_sentences = []
                for sentence in sentences:
                    preprocessed_sentences = []
                    tokens = parse(sentense)
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = len(word2idx)
                        preprocessed_sentence.append(word2idx[token])
                    preprocessed_sentences.append(preprocessed_sentence)
            with open(preprocessed_scenarios_dir / scenario.name, 'w') as f:
                json.dump(preprocessed_sentences, f)
    with open(dictionary_dir / 'word2idx.json', 'w') as f:
        json.dump(word2idx, f)

def parse(sentence):
    words = m.parse(sentence).split("\n")[:-1]
    return map(lambda x: x.split("\t")[0], tokens)

main()
