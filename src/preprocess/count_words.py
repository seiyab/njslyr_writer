from pathlib import Path
import json
from collections import Counter

preprocessed_dir = Path('/preprocessed')

def main():
    c = Counter()
    for jsonfile in preprocessed_dir.glob('*.json'):
        with open(jsonfile, 'r') as f:
            c.update(sum(json.load(f), []))
    with open(preprocessed_dir / 'dictionaries' / 'bag_of_words.json') as f:
        json.dump(dict(c), f)

if __name__ == "__main__":
    main()
