#! /usr/bin/env python3

from word2vec import Dictionary, Embedding, Corpus, train as train_emb
from gensim.models import word2vec
import joblib
import logging, sys
import click

logger = logging.getLogger(__name__)
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'

@click.group()
def cli():
    pass

@cli.command()
@click.argument('corpus', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--min-count', type=int, default=5)
def build_dic(corpus, output, min_count):
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger.info('Build dictionary')
    dic = Dictionary.build(corpus, min_count=min_count)

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
@click.option('--symmetric/--no-symmetric', default=False)
@click.option('--init_alpha', type=float, default=0.025)
@click.option('--min_alpha', type=float, default=0.0001)
@click.option('--neg-power', type=float, default=3/4)
@click.option('--workers', type=int, default=5)
def train(corpus, dictionary, output, iteration, symmetric, **kwargs):
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger.info('Load dictionary')

    if symmetric:
        kwargs['ctx_negative'] = True
    else:
        kwargs['ctx_negative'] = False

    dic = Dictionary.load(dictionary)

    corpus = Corpus(dic, corpus, iteration)

    emb = train_emb(dic, corpus, **kwargs)

    logger.info('Save')

    emb.save(output)
    emb.save_text(output + '.w2v')

@cli.command()
@click.argument('corpus', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--min-count', type=int, default=5)
@click.option('--iteration', type=int, default=5)
@click.option('--dim', type=int, default=100)
@click.option('--window', type=int, default=5)
@click.option('--negative', type=int, default=10)
@click.option('--init_alpha', type=float, default=0.025)
@click.option('--min_alpha', type=float, default=0.0001)
@click.option('--neg-power', type=float, default=3/4)
@click.option('--workers', type=int, default=5)
def train_gensim(corpus, output, min_count, iteration, dim, window, negative, init_alpha, min_alpha, neg_power, workers):
    model = word2vec.Word2Vec(corpus_file=corpus, sg=1, size=dim, 
            alpha=init_alpha, min_alpha=min_alpha, 
            window=window, min_count=min_count, 
            negative=negative, ns_exponent=neg_power, iter=iteration,
            workers=workers)
    
    model.save(output)


if __name__ == '__main__':
    cli()
