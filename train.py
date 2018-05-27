from word2vec import data, train
import joblib
import logging, sys
import click

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)

@click.command()
@click.argument('corpus', type=click.Path(exists=True))
@click.argument('dictionary', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def main(corpus, dictionary, output):
    logger.info('Load dictionary')
    dic = data.Dictionary.load(dictionary)

    logger.info('Load corpus')
    corpus = data.Corpus(dic, corpus, 5)

    logger.info('Start training')
    emb = train.train(dic, corpus, dim=100, init_alpha=0.025, min_alpha=0.0001, window=5, negative=5, neg_power=3/4)

    logger.info('Save')

    emb.save(args.output)
    emb.save_text(args.output + '.w2v')

if __name__ == '__main__':
    main()
