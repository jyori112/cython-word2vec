#! /usr/bin/env python3

import click
from word2vec import Embedding, evaluation
import logging

logger = logging.getLogger(__name__)

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'

def format_result(result):
    def f(value):
        if isinstance(value, float):
            return '{:.3f}'.format(value)
        else:
            return value

    return '; '.join('{}={}'.format(key, f(value)) for key, value in result.items())

def evalute_wordsim(key2vec, wordsim_path):
    # Gensim evaluate_word_pairs() produce FutureWarning, Ignore it
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    pearson, spearman, oov = key2vec.evaluate_word_pairs(wordsim_path)
    pearson = pearson[0]
    spearman = spearman[0]
    covarage = (100-oov)/100

    return dict(pearson=pearson, spearman=spearman, covarage=covarage)

def evaluate_anology(key2vec, analogy_path):
    # Gensim evaluate_word_pairs() produce FutureWarning, Ignore it
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    acc, sections = key2vec.evaluate_word_analogies(analogy_path)

    return dict(acc=acc)
    

@cli.command()
@click.argument('embedding', type=click.Path(exists=True))
@click.argument('wordsim', type=click.Path(exists=True))
@click.option('--emb-type', type=click.Choice(['all', 'trg', 'ctx']), default='all')
def eval_wordsim(embedding, wordsim, emb_type):
    # Gensim evaluate_word_pairs() produce FutureWarning, Ignore it
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    emb = Embedding.load(embedding)

    if emb_type == 'all':
        emb_types = ['trg', 'ctx']
    else:
        emb_types = [emb_type]

    for emb_type in emb_types:
        if emb_type == 'trg':
            emb.trg()
        elif emb_type == 'ctx':
            emb.ctx()

        k2v = emb.to_gensim()
        result = evalute_wordsim(k2v, wordsim)
        result['emb-type'] = emb_type
        print(format_result(result))

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

    evaluate_wordsim(k2v, wordsim, out_format)

if __name__ ==  '__main__':
    cli()
