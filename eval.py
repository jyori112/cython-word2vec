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
@click.option('--emb-type', type=click.Choice(['trg', 'ctx']))
def eval_wordsim(embedding, wordsim, emb_type):
    emb = Embedding.load(embedding)

    wordsim = evaluation.WordSim.load(emb._dic, wordsim, sep='\t')

    if emb_type is None:
        emb_types = ['trg', 'ctx']
    else:
        emb_types = [emb_type]

    for emb_type in emb_types:
        if emb_type == 'trg':
            emb.trg()
        elif emb_type == 'ctx':
            emb.ctx()

        score = wordsim.evaluate(emb)

        logger.info('Type={}; Spearman={:.3f}'.format(emb_type, score))

if __name__ ==  '__main__':
    cli()
