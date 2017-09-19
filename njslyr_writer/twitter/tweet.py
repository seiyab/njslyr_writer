#!/opt/conda/bin/python3

from pathlib import Path
import numpy
from njslyr_writer.twitter import twpy
from njslyr_writer.generate import write as w

models_dir = Path('/models')

def main():
    generator = np.random.choice(list(models_dir.glob('generators/*.pkl')))
    sentence = w.write(generator)
    twpy.api.update_status(sentence[:140])

if __name__ == '__main__':
    main()
