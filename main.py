#! /usr/bin/env python3

from word2vec import Dictionary, Corpus, train as train_emb
import joblib
import logging, sys
import click

logger = logging.getLogger(__name__)

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
@click.option('--iteration', type=int, default=5)
@click.option('--dim', type=int, default=100)
@click.option('--window', type=int, default=5)
@click.option('--negative', type=int, default=10)
@click.option('--init_alpha', type=float, default=0.025)
@click.option('--min_alpha', type=float, default=0.0001)
@click.option('--neg-power', type=float, default=3/4)
def train(corpus, dictionary, output, iteration, **kwargs):
    logger.info('Load dictionary')
    dic = Dictionary.load(dictionary)

    corpus = Corpus(dic, corpus, iteration)

    emb = train_emb(dic, corpus, **kwargs)

    logger.info('Save')

    emb.save(output)
    emb.save_text(output + '.w2v')

if __name__ == '__main__':
    cli()
