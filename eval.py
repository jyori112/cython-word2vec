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
@click.option('--type', type=click.Choice(['trg', 'ctx']), default='trg')
def eval_wordsim(embedding, wordsim, type):
    emb = Embedding.load(embedding)

    if type == 'trg':
        emb.trg()
    elif type == 'ctx':
        emb.ctx()

    wordsim = evaluation.WordSim.load(emb._dic, wordsim, sep='\t')
    score = wordsim.evaluate(emb)

    logger.info('Spearman={:.3f}'.format(score))

if __name__ ==  '__main__':
    cli()
