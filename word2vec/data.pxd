cimport numpy as np

cdef class Word:
    cdef public int index
    cdef public int count
    cdef public str text

cdef class Vocab:
    cdef _id2word
    cdef _text2word
    cdef np.ndarray _neg_table

    cpdef Word text2word(self, str text)
    cpdef Word id2word(self, int idx)
    cpdef list encode(self, list tokens)
    cpdef list decode(self, list indexes)
    cpdef build_neg_table(self, float power, int table_size)
    cpdef Word sample_negative(self)

cdef class Dictionary:
    cdef src_vocab, trg_vocab
    cdef dict trans

    cpdef list get_trans(self, Word word)

cdef class Corpus:
    cdef _vocab
    cdef _path
    cdef _n_lines, _processed_line, _epoch

cdef class Embedding:
    cdef public np.ndarray trg
    cdef public np.ndarray ctx
    cdef public np.ndarray trg_nrm
    cdef public Vocab vocab

    cpdef np.ndarray get_vec(self, Word word)
    cpdef list get_similar_by_vec(self, np.ndarray vec)
    cpdef list get_similar(self, Word word)

