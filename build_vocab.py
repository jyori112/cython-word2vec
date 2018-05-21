from word2vec import data, train
import argparse
import joblib
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str)
    parser.add_argument('output', type=str)
    
    args = parser.parse_args()

    logger.info('Build vocab')
    vocab = data.Vocab.build(args.corpus)

    logger.info('Saving vocab')
    vocab.save(args.output)

if __name__ == '__main__':
    main()
