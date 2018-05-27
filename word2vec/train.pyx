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
    cdef dict _params

    def __init__(self, params):
        self._params = params

        for key, value in self._params.items():
            setattr(self, key, value)

# Global Variables
cdef Dictionary _dic
cdef int _n_line
cdef Corpus _corpus
cdef float32_t [:, :] _trg
cdef float32_t [:, :] _ctx
cdef int32_t [:] _neg_table
cdef float32_t [:] _exp_table
cdef Parameters _param
cdef object _alpha
cdef object _line_counter
cdef float32_t [:] _work

def train(Dictionary dic, Corpus corpus, **kwargs):
    param = Parameters(kwargs)

    dic = dic
    n_line = len(corpus)

    line_counter = Value(c_uint64, 0)
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
        line_counter,
    )
    with Pool(cpu_count(), initializer=init_work, initargs=init_args) as p:
        processed_tokens = 0
        with tqdm(total=n_line, mininterval=0.5) as bar:
            for i, n_tokens in enumerate(p.imap_unordered(train_line, corpus, chunksize=30)):
                bar.update(1)

    return Embedding(dic, trg, ctx)

def init_work(dic, n_line, trg_shared, ctx_shared, neg_table, exp_table, param, alpha, line_counter):
    global _dic, _n_line, _trg, _ctx, _neg_table, _exp_table, _param, _alpha, _line_counter, _work, _dim

    _dic = dic
    _n_line = n_line

    _neg_table = neg_table
    _exp_table = exp_table

    _param = param
    _alpha = alpha
    _line_counter = line_counter

    _trg = np.frombuffer(trg_shared, dtype=np.float32).reshape(len(dic), _param.dim)
    _ctx = np.frombuffer(ctx_shared, dtype=np.float32).reshape(len(dic), _param.dim)
    _work = np.zeros(_param.dim, dtype=np.float32)

def train_line(list line):
    """
        line: sequence of Words
    """
    cdef int center_pos, context_pos, start, end
    cdef Word center
    cdef Word context
    cdef float p
    
    for center_pos in range(len(line)):
        start = max(0, center_pos - _param.window)
        end = min(len(line), center_pos + _param.window + 1)
        for context_pos in range(start, end):
            center = line[center_pos]
            context = line[context_pos]
            if center.index != context.index:
                _train(center, context)

    return len(line)

    with _line_counter.get_lock():
        _line_counter.value += 1
        p = 1.0 - float(_line_counter.value) / _n_line
        _alpha.value = max(_param.min_alpha, p * _param.init_alpha)
    
cdef inline void _train(Word trg, Word ctx):
    cdef float32_t label, f, g, f_dot
    cdef int one = 1
    cdef float32_t onef = <float32_t>1.0
    cdef int dim = _param.dim

    memset(&_work[0], 0, _param.dim * cython.sizeof(float32_t))

    for d in range(_param.negative + 1):
        if d == 0:
            # This is normal
            ctx_index = ctx.index
            label = 1.0
        else:
            # Negative Sample
            neg_index = np.random.randint(0, NEG_TABLE_SIZE)
            ctx_index = _neg_table[neg_index]
            label = 0.0

        f_dot = <float32_t>(blas.sdot(&dim, &_trg[trg.index, 0], &one, &_ctx[ctx_index, 0], &one))

        if f_dot >= MAX_EXP or f_dot <= -MAX_EXP:
            # What if the f_dot is too high for wrong direction?
            continue

        f = _exp_table[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * _alpha.value

        blas.saxpy(&dim, &g, &_ctx[ctx_index, 0], &one, &_work[0], &one)
        blas.saxpy(&dim, &g, &_trg[trg.index, 0], &one, &_ctx[ctx_index, 0], &one)

    blas.saxpy(&dim, &onef, &_work[0], &one, &_trg[trg.index, 0], &one)
        



