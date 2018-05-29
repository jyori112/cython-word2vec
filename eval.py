#! /usr/bin/env python3

import click
from word2vec import Embedding, evaluation
import logging

logger = logging.getLogger(__name__)

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('embedding', type=click.Path(exists=True))
@click.argument('wordsim', type=click.Path(exists=True))
def eval_wordsim(embedding, wordsim):
    emb = Embedding.load(embedding)
    wordsim = evaluation.WordSim.load(emb.dic, wordsim, sep='\t')
    score = wordsim.evaluate(emb)

    logger.info('Spearman={:.3f}'.format(score))

if __name__ ==  '__main__':
    cli()
