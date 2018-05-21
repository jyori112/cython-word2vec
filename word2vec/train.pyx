import numpy as np
cimport numpy as np
from ctypes import c_float, c_uint64
cimport cython
import scipy.stats, scipy.spatial.distance
from scipy.linalg cimport cython_blas as blas
from libc.math cimport exp
from multiprocessing.pool import Pool
from multiprocessing import RawArray, RawValue, Value
from libc.string cimport memset
import traceback
from .data cimport Word, Vocab, Corpus, Embedding
import logging, sys, os

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)

# Global Variables
cdef Vocab _vocab
cdef Corpus _corpus
cdef np.ndarray _trg
cdef np.ndarray _ctx
cdef np.ndarray _work
cdef np.ndarray _context_vector
cdef int _dim, _window, _negative
cdef object _alpha
cdef object _line_counter

cdef int ONE = 1
cdef np.float32_t ONEF = <np.float32_t>1.0
cdef np.float32_t MAX_EXP = <np.float32_t>6.0

def train_line(list line):
    """
        line: sequence of Words
    """
    cdef int center_pos, start, end
    cdef Word center
    cdef list contexts
    cdef float p
    
    try:
        for center_pos in range(len(line)):
            start = max(0, center_pos - _window)
            end = min(len(line), center_pos + _window + 1)
            center = line[center_pos]
            contexts = line[start:end]

            _train(center, contexts)
    except:
        print(traceback.format_exc())
        raise

    with _line_counter.get_lock():
        _line_counter.value += 1
        p = 1.0 - float(_line_counter.value) / _corpus.n_lines
        _alpha.value = max(0.0001, p * 0.025)
    
cdef inline void _train(Word trg, list ctxs):
    cdef np.float32_t f, g, label
    cdef int trg_index, trg_row, ctx_row, best_row, row, tran_idx
    cdef np.float32_t best_score, nrm, dot, trg_nrm
    cdef Word negative_sample

    cdef np.float32_t *trg_ = <np.float32_t *>(np.PyArray_DATA(_trg))
    cdef np.float32_t *ctx_ = <np.float32_t *>(np.PyArray_DATA(_ctx))
    cdef np.float32_t *work_ = <np.float32_t *>(np.PyArray_DATA(_work))
    cdef np.float32_t *context_vector_ = <np.float32_t *>(np.PyArray_DATA(_context_vector))

    try:

        # Initialize Context vector
        memset(context_vector_, 0, _dim * cython.sizeof(np.float32_t))

        # Compute Context Vector
        for ctx in ctxs:
            ctx_row = <unsigned long>(ctx.index * _dim)
            blas.saxpy(&_dim, &ONEF, &ctx_[ctx_row], &ONE, context_vector_, &ONE)

        # Initialize Work place
        memset(work_, 0, _dim * cython.sizeof(np.float32_t))

        for d in range(_negative + 1):
            if d == 0:
                trg_index = trg.index
                label = 1.0
            else:
                #print("Negative Sampling")
                negative_sample = _vocab.sample_negative()
                #print(negative_sample)
                trg_index = negative_sample.index
                label = 0.0

            trg_row = <unsigned long>(trg_index * _dim)

            f = <np.float32_t>blas.sdot(&_dim, &trg_[trg_row], &ONE, context_vector_, &ONE)

            if f > MAX_EXP:
                f = 1.0
                g = (label-1.0) * _alpha.value
            elif f < -MAX_EXP:
                f = 0.0
                g = (label-0.0) * _alpha.value
            else:
                f = 1.0 / (1.0 + exp(-f))
                g = (label-f) * _alpha.value

            #print(label, f, g)
            blas.saxpy(&_dim, &g, &trg_[trg_row], &ONE, work_, &ONE)
            blas.saxpy(&_dim, &g, context_vector_, &ONE, &trg_[trg_row], &ONE)
        
        # Update context vector
        for ctx in ctxs:
            ctx_row = <unsigned long>(ctx.index * _dim)
            blas.saxpy(&_dim, &ONEF, work_, &ONE, &ctx_[ctx_row], &ONE)
    except:
        print(traceback.format_exc())
        raise

def train(vocab, corpus, wordsim=None, dim=100, window=5, negative=5):
    global _vocab, _corpus
    global _trg, _ctx
    global _dim, _window, _negative, _alpha
    global _context_vector, _work
    global _alpha, _line_counter

    logger.info('--- Training')

    _vocab = vocab
    _corpus = corpus

    _dim = dim
    _window = window
    _negative = negative
    _line_counter = Value(c_uint64, 0)
    _alpha = RawValue(c_float, 0.025)

    _context_vector = np.zeros(shape=(dim, ), dtype=np.float32)
    _work = np.zeros(shape=(dim, ), dtype=np.float32)

    logger.info('Initialize embeddings')
    trg_shared = RawArray(c_float, len(vocab) * dim)
    ctx_shared = RawArray(c_float, len(vocab) * dim)

    _trg = np.frombuffer(trg_shared, dtype=np.float32)
    _ctx = np.frombuffer(ctx_shared, dtype=np.float32)

    _trg = _trg.reshape(len(vocab), dim)
    _ctx = _ctx.reshape(len(vocab), dim)

    _trg[:] = (np.random.rand(len(vocab), dim).astype(np.float32) - 0.5) / dim
    _ctx[:] = np.zeros((len(vocab), dim), dtype=np.float32)

    logger.info('Start training')
    with Pool(15, initializer=init_work, initargs=(trg_shared, ctx_shared)) as p:
        for i, _ in enumerate(p.imap_unordered(train_line, corpus, chunksize=30)):
            if i % 100000 == 0:
                if wordsim:
                    spearson, coverage = evaluate(wordsim)
                    logger.info('prog={:.2f}%; spearman={:.3f}; coverage={:.3f}; alpha={:.3f}'.format(
                        i/corpus.n_lines*100, spearson, coverage, _alpha.value))
                else:
                    logger.info('prg={:.2f}%; alpha={:.3f}'.format(i/corpus.n_lines * 100, _alpha.value))

    return Embedding(_vocab, _trg, _ctx)

def init_work(trg_shared, ctx_shared):
    global _trg, _ctx

    _trg = np.frombuffer(trg_shared, dtype=np.float32)
    _ctx = np.frombuffer(ctx_shared, dtype=np.float32)
    
    _trg = _trg.reshape(len(_vocab), _dim)
    _ctx = _ctx.reshape(len(_vocab), _dim)

def evaluate(wordsim):
    model = []
    gold = []
    oov = 0

    for word1, word2, sim in wordsim:
        if word1.is_UNK or word2.is_UNK:
            oov += 1

        emb1 = _trg[word1.index]
        emb2 = _trg[word2.index]

        score = 1-scipy.spatial.distance.cosine(emb1, emb2)
        model.append(score)
        gold.append(sim)

    return scipy.stats.spearmanr(model, gold)[0], len(model) / (len(model) + oov)
