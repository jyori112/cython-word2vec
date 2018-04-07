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
    parser.add_argument('vocab', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--dim', '-d', type=int, default=100)
    parser.add_argument('--window', '-w', type=int, default=5)
    parser.add_argument('--negative', '-n', type=int, default=5)
    parser.add_argument('--n_lines', type=int, default=10000000)
    parser.add_argument('--neg_table_size', type=int, default=10000000)
    parser.add_argument('--neg_power', type=float, default=3/4)
    parser.add_argument('--wordsim', type=str)

    args = parser.parse_args()


    logger.info('Load vocab')
    vocab = data.Vocab.load(args.vocab)

    logger.info('Build neg table')
    vocab.build_neg_table(args.neg_power, args.neg_table_size)

    corpus = data.Corpus(vocab, args.corpus, args.n_lines)

    if args.wordsim:
        wordsim  = vocab.load_wordsim(args.wordsim)
    else:
        wordsim = None

    logger.info('Start training')
    emb = train.train(vocab, corpus, wordsim=wordsim,
            dim=args.dim, window=args.window, negative=args.negative)

    logger.info('Save')

    emb.save(args.output)
    emb.save_text(args.output + '.w2v')

if __name__ == '__main__':
    main()
