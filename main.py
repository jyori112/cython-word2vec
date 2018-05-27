#! /usr/bin/python

from word2vec import Dictionary, Corpus, train
import joblib
import logging, sys
import click

loggr = logging.getLogger(__name__)

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@cli.command()
@click.argument('corpus', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def build_dic(corpus, output):
    logger.info('Build dictionary')
    dic = Dictionary.build(corpus)

    logger.info('Saving dictionary')
    dic.save(output)

@cli.command()
@click.argument('corpus', type=click.Path(exists=True))
@click.argument('dictionary', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def main(corpus, dictionary, output):
    logger.info('Load dictionary')
    dic = Dictionary.load(dictionary)

    logger.info('Load corpus')
    corpus = Corpus(dic, corpus, 5)

    logger.info('Start training')
    emb = train(dic, corpus, dim=100, init_alpha=0.025, min_alpha=0.0001, window=5, negative=5, neg_power=3/4)

    logger.info('Save')

    emb.save(args.output)
    emb.save_text(args.output + '.w2v')

if __name__ == '__main__':
    cli()
