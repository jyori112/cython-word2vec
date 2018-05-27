cimport numpy as np

cdef class Word:
    cdef public int index
    cdef public int count
    cdef public str text

cdef class Dictionary:
    cdef dict _index2word
    cdef dict _text2word
    cdef int n_lines

    cpdef Word text2word(self, str text)
    cpdef Word index2word(self, int index)
    cpdef list encode(self, list tokens)
    cpdef list decode(self, list indexes)

    cpdef build_neg_table(self, float power, int table_size)

cdef class Corpus:
    cdef public Dictionary dic
    cdef public str path
    cdef public int n_epoch, epoch
    cdef public int n_words, processed_words

cdef class Embedding:
    cdef public np.ndarray trg
    cdef public np.ndarray ctx
    cdef public np.ndarray trg_nrm
    cdef public Dictionary dic

    cpdef np.ndarray get_vec(self, Word word)
    cpdef list get_similar_by_vec(self, np.ndarray vec, int count)
    cpdef list get_similar(self, Word word, int count)

