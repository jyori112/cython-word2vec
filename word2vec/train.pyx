import numpy as np
cimport numpy as np
from ctypes import c_float, c_uint64
from libc.stdint cimport int32_t
cimport cython
import scipy.stats, scipy.spatial.distance
from scipy.linalg cimport cython_blas as blas
from libc.math cimport exp
from multiprocessing.pool import Pool
from multiprocessing import RawArray, RawValue, Value, cpu_count
from libc.string cimport memset
import traceback
from .data cimport Word, Dictionary, Corpus, Embedding
from .random cimport seed, randint_c
import logging, sys, os
from tqdm import tqdm, tqdm_notebook

ctypedef np.float32_t float32_t

logger = logging.getLogger(__name__)

DEF MAX_EXP = 6
DEF EXP_TABLE_SIZE = 1000
DEF NEG_TABLE_SIZE = 100000000

cdef class Parameters:
    cdef public int32_t dim
    cdef public float32_t init_alpha
    cdef public float32_t min_alpha
    cdef public int32_t window
    cdef public int32_t negative
    cdef public float32_t neg_power
    cdef public int32_t workers
    cdef dict _params

    def __init__(self, params):
        self._params = params

        for key, value in self._params.items():
            setattr(self, key, value)

# Global Variables
cdef Dictionary _dic
cdef float  _n_line
cdef Corpus _corpus
cdef float32_t [:, :] _trg
cdef float32_t [:, :] _ctx
cdef int32_t [:] _neg_table
cdef float32_t [:] _exp_table
cdef Parameters _param
cdef object _alpha
cdef float32_t [:] _trg_grad, _ctx_grad

def train(Dictionary dic, Corpus corpus, **kwargs):
    param = Parameters(kwargs)

    dic = dic
    n_line = len(corpus)

    alpha = RawValue(c_float, param.init_alpha)

    logger.info("Building Negative Sampling Table")
    neg_table = dic.build_neg_table(param.neg_power, NEG_TABLE_SIZE)

    logger.info("Buiding Exponent Table for Faster Computation")
    exp_table = RawArray(c_float, EXP_TABLE_SIZE)
    for i in range(EXP_TABLE_SIZE):
        exp_table[i] = <float32_t>exp((i / <float32_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        exp_table[i] = <float32_t>(exp_table[i] / (exp_table[i] + 1))

    logger.info("Initializing Embeddings")
    trg_shared = RawArray(c_float, len(dic) * param.dim)
    ctx_shared = RawArray(c_float, len(dic) * param.dim)

    trg = np.frombuffer(trg_shared, dtype=np.float32).reshape(len(dic), param.dim)
    ctx = np.frombuffer(ctx_shared, dtype=np.float32).reshape(len(dic), param.dim)

    trg[:] = (np.random.rand(len(dic), param.dim) - 0.5) / param.dim
    ctx[:] = np.zeros((len(dic), param.dim))

    logger.info('Start training')
    init_args = (
        dic,
        n_line,
        trg_shared, ctx_shared,
        neg_table,
        exp_table,
        param,
        alpha,
    )

    poolsize = param.workers
    #poolsize = 1
    with Pool(poolsize, initializer=init_work, initargs=init_args) as p:
        with tqdm(total=n_line, mininterval=0.5) as bar:
            for _ in p.imap_unordered(train_line, enumerate(corpus.indexes()), chunksize=100):
                bar.update(1)
                #tqdm.write("Alpha: {:.5}".format(alpha.value))

    return Embedding(dic, trg, ctx)

def init_work(dic, n_line, trg_shared, ctx_shared, neg_table, exp_table, param, alpha):
    global _dic, _n_line, _trg, _ctx, _neg_table, _exp_table, _param, _alpha, _dim, _trg_grad, _ctx_grad

    _dic = dic
    _n_line = n_line

    _neg_table = neg_table
    _exp_table = exp_table

    _param = param
    _alpha = alpha

    _trg = np.frombuffer(trg_shared, dtype=np.float32).reshape(len(dic), _param.dim)
    _ctx = np.frombuffer(ctx_shared, dtype=np.float32).reshape(len(dic), _param.dim)
    _trg_grad = np.zeros(_param.dim, dtype=np.float32)
    _ctx_grad = np.zeros(_param.dim, dtype=np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def train_line(tuple args):
    """
        line: sequence of Words
    """
    cdef int32_t center_pos, context_pos, start, end
    cdef int32_t center
    cdef int32_t context
    cdef float32_t p
    cdef int line_n
    cdef list line
    cdef int n_tokens
    cdef float32_t alpha_

    line_n, line = args
    
    n_tokens = len(line)

    alpha_ = _alpha.value


    for center_pos in xrange(n_tokens):
        start = max(0, center_pos - _param.window)
        end = min(n_tokens, center_pos + _param.window + 1)
        for context_pos in xrange(start, end):
            center = line[center_pos]
            context = line[context_pos]
            if center_pos != context_pos:
                _train(center, context, alpha_)

    _alpha.value = max(_param.min_alpha, _param.init_alpha * (1.0-line_n/_n_line))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void _train(int32_t trg, int32_t ctx, float32_t alpha):
    cdef float32_t f, g, f_dot
    cdef int one = 1
    cdef float32_t onef = <float32_t>1.0
    cdef int dim = _param.dim

    memset(&_trg_grad[0], 0, _param.dim * cython.sizeof(float32_t))
    memset(&_ctx_grad[0], 0, _param.dim * cython.sizeof(float32_t))

    # This is normal
    f_dot = <float32_t>(blas.sdot(&dim, &_trg[trg, 0], &one, &_ctx[ctx, 0], &one))

    if -MAX_EXP < f_dot and f_dot < MAX_EXP:
        f = _exp_table[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1.0 - f) * alpha

        blas.saxpy(&dim, &g, &_ctx[ctx, 0], &one, &_trg_grad[0], &one)
        blas.saxpy(&dim, &g, &_trg[trg, 0], &one, &_ctx[ctx, 0], &one)

    for d in range(_param.negative):
        # Negative Sample
        ctx_neg = _neg_table[randint_c() % NEG_TABLE_SIZE]

        f_dot = <float32_t>(blas.sdot(&dim, &_trg[trg, 0], &one, &_ctx[ctx_neg, 0], &one))

        if -MAX_EXP < f_dot and f_dot < MAX_EXP:
            f = _exp_table[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            g = - f * alpha

            blas.saxpy(&dim, &g, &_ctx[ctx_neg, 0], &one, &_trg_grad[0], &one)
            blas.saxpy(&dim, &g, &_trg[trg, 0], &one, &_ctx[ctx_neg, 0], &one)

    blas.saxpy(&dim, &onef, &_trg_grad[0], &one, &_trg[trg, 0], &one)
