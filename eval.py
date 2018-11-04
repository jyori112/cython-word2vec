#! /usr/bin/env python3

import click
from word2vec import Embedding, evaluation
import logging

logger = logging.getLogger(__name__)

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'

def evalute_wordsim(key2vec, wordsim_path, out_format='log', emb_type=None):
    # Gensim evaluate_word_pairs() produce FutureWarning, Ignore it
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    pearson, spearman, oov = key2vec.evaluate_word_pairs(wordsim_path)
    pearson = pearson[0]
    spearman = spearman[0]
    covarage = (100-oov)/100

    if out_format == 'log':
        if emb_type is not None:
            print('emb-type={}; pearson={:.3f}; spearman={:.3f}; coverage={:.3f}'.format(
                emb_type, pearson, spearman, covarage))
        else:
            print('pearson={:.3f}; spearman={:.3f}; coverage={:.3f}'.format(
                pearson, spearman, covarage))
    elif out_format in ('csv', 'tsv'):
        sep = ',' if out_format == 'csv' else '\t'
        if emb_type is not None:
            print(emb_type, pearson, spearman, covarage, sep=sep)
        else:
            print(pearson, spearman, covarage, sep=sep)


@cli.command()
@click.argument('embedding', type=click.Path(exists=True))
@click.argument('wordsim', type=click.Path(exists=True))
@click.option('--emb-type', type=click.Choice(['trg', 'ctx']))
@click.option('--out-format', type=click.Choice(['csv', 'tsv', 'log']), default='log')
@click.option('--header/--no-header', default=True)
def eval_wordsim(embedding, wordsim, emb_type, out_format, header):
    # Gensim evaluate_word_pairs() produce FutureWarning, Ignore it
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    emb = Embedding.load(embedding)

    if emb_type is None:
        emb_types = ['trg', 'ctx']
    else:
        emb_types = [emb_type]


    if header and out_format in ('csv', 'tsv'):
        sep = ',' if out_format == 'csv' else '\t'
        print('emb-type', 'pearson', 'spearman', 'coverage', sep=sep)

    for emb_type in emb_types:
        if emb_type == 'trg':
            emb.trg()
        elif emb_type == 'ctx':
            emb.ctx()

        k2v = emb.to_gensim()
        evalute_wordsim(k2v, wordsim, out_format, emb_type=emb_type)

@cli.command()
@click.argument('embedding', type=click.Path(exists=True))
@click.argument('wordsim', type=click.Path(exists=True))
@click.option('--out-format', type=click.Choice(['csv', 'tsv', 'log']), default='log')
@click.option('--header/--no-header', default=True)
def eval_wordsim_gensim(embeddings, wordsim, out_format, header):
    k2v = models.keyedvectors.WordEmbeddingsKeyedVectors.load(embeddings)

    if header and out_format in ('csv', 'tsv'):
        sep = ',' if out_format == 'csv' else '\t'
        print('pearson', 'spearman', 'coverage', sep=sep)

    evalute_gensim_with_wordsim(k2v, wordsim, out_format)

if __name__ ==  '__main__':
    cli()
