#! /usr/bin/env python3

import click
from word2vec import Embedding, evaluation
import logging
from gensim import models

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
@click.argument('eval-data', type=click.Path(exists=True))
@click.option('--emb-type', type=click.Choice(['all', 'trg', 'ctx']), default='all')
@click.option('--similarity/--analogy', default=True)
def eval(embedding, eval_data, emb_type, similarity):
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
        if similarity:
            result = evalute_wordsim(k2v, eval_data)
        else:
            result = evaluate_anology(k2v, eval_data)
        result['emb-type'] = emb_type
        print(format_result(result))

@cli.command()
@click.argument('embedding', type=click.Path(exists=True))
@click.argument('eval-data', type=click.Path(exists=True))
@click.option('--similarity/--analogy', default=True)
def eval_gensim(embedding, eval_data, similarity):
    k2v = models.word2vec.Word2Vec.load(embedding).wv

    if similarity:
        result = evalute_wordsim(k2v, eval_data)
    else:
        result = evaluate_anology(k2v, eval_data)

    print(format_result(result))

if __name__ ==  '__main__':
    cli()
