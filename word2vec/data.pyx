from collections import Counter, defaultdict
import joblib
from ctypes import c_float, c_int32, c_uint32, c_uint64
from libc.stdint cimport int32_t, uint32_t
import multiprocessing
import numpy as np
cimport numpy as np
import scipy.stats
import scipy.spatial.distance

cdef class Word:
    def __init__(self, int index, str text, int count):
        self.index = index
        self.text = text
        self.count = count

    @property
    def is_UNK(self):
        return self.index == 0

    def __repr__(self):
        return "<Word {} />".format(self.text)

cdef class Dictionary:
    def __init__(self, int n_lines, list word_list):
        self.n_lines = n_lines
        self._index2word = {word.index: word for word in word_list}
        self._text2word = {word.text: word for word in word_list}

    @property
    def UNK(self):
        return self._text2word['__UNK__']

    def total_word_count(self):
        return sum(word.count for word in self)

    cpdef Word text2word(self, str text):
        if text in self._text2word:
            return self._text2word[text]
        else:
            return self.UNK

    cpdef Word index2word(self, int idx):
        return self._index2word[idx]

    cpdef list encode(self, list tokens):
        return [self.text2word(token) for token in tokens]

    cpdef list decode(self, list indexes):
        return [self.index2word(idx) for idx in indexes]

    def __len__(self):
        return len(self._index2word)

    def __iter__(self):
        for index in range(len(self)):
            yield self._index2word[index]

    cpdef build_neg_table(self, float power, int table_size):
        cdef list items
        cdef float items_pow, cur
        cdef int index, table_index

        neg_table = multiprocessing.RawArray(c_int32, table_size)
        items = list(self)
        items_pow = float(sum([item.count ** power for item in self]))

        index = 0
        cur = items[index].count ** power / items_pow

        for table_index in xrange(table_size):
            neg_table[table_index] = items[index].index
            if float(table_index) / table_size > cur:
                if index < len(items) - 1:
                    index += 1
                cur += items[index].count ** power / items_pow

        return neg_table

    @staticmethod
    def build(path):
        word_count = Counter()
        n_lines = 0
        with open(path) as f:
            for line in f:
                tokens = line.split()
                word_count.update(tokens)
                n_lines += 1

        word_list = [(text, count) for text, count in word_count.items() if count >= 5]
        word_list = sorted(word_list, key=lambda x: x[1], reverse=True)
        word_list = [Word(idx, text, count) for idx, (text, count) in enumerate(word_list)]

        word_list.append(Word(len(word_list), '__UNK__', 0))

        return Dictionary(n_lines, word_list)

    def save(self, path):
        word_list = list(self._index2word.values())
        joblib.dump(dict(word_list=word_list, n_lines=self.n_lines), path)

    @staticmethod
    def load(path):
        loader = joblib.load(path)
        return Dictionary(loader['n_lines'], loader['word_list'])

cdef class Corpus:
    def __init__(self, Dictionary dic, str path, int n_epoch):
        self.dic = dic
        self.path = path
        self.n_epoch = n_epoch

    def __len__(self):
        return self.dic.n_lines * self.n_epoch

    def __iter__(self):
        cdef str line
        cdef list tokens

        with open(self.path) as f:
            for self.epoch in range(self.n_epoch):
                # Reset epoch
                f.seek(0)

                # Iterate in file
                for line in f:
                    tokens = self.dic.encode(line.split())
                    yield tokens

    def indexes(self):
        cdef str line
        cdef list tokens

        with open(self.path) as f:
            for self.epoch in xrange(self.n_epoch):
                f.seek(0)

                for line in f:
                    tokens = self.dic.encode(line.split())
                    yield [w.index for w in tokens]

cdef class Embedding:
    def __init__(self, Dictionary dic, np.ndarray trg, np.ndarray ctx):
        self._dic = dic
        self._trg = trg
        self._ctx = ctx
        self._trg_nrm = self._trg / np.linalg.norm(self._trg, axis=1)[:, None]
        self._ctx_nrm = self._ctx / np.linalg.norm(self._ctx, axis=1)[:, None]
        self._mode = 'trg'

    def ctx(self):
        self._mode = 'ctx'

    def trg(self):
        self._mode = 'trg'

    @property
    def matrix(self):
        if self._mode == 'trg':
            return self._trg
        elif self._mode == 'ctx':
            return self._ctx
        else:
            raise Exception('Unknown mode')

    @property
    def norm_matrix(self):
        if self._mode == 'trg':
            return self._trg_nrm
        elif self._mode == 'ctx':
            return self._ctx_nrm
        else:
            raise Exception('Unknown mode')

    cpdef np.ndarray get_vec(self, Word word):
        return self.matrix[word.index]

    cpdef list get_similar_by_vec(self, np.ndarray vec, int count):
        score = np.dot(self.norm_matrix, vec) / np.linalg.norm(vec)
        rank = np.argsort(-score)[:count]
        return [(self.dic.index2word(index), score[index]) for index in rank]
    
    cpdef list get_similar(self, Word word, int count):
        return self.get_similar_by_vec(self.get_vec(word), count)

    def save(self, path):
        joblib.dump(dict(dic=self._dic, trg=self._trg, ctx=self._ctx), path)

    def save_text(self, path):
        with open(path, 'w') as f:
            f.write('{} {}\n'.format(self.matrix.shape[0], self.matrix.shape[1]))
            for word in self.dic:
                vec_str = ' '.join('{:.6f}'.format(v) for v in self.get_vec(word))
                f.write('{} {}\n'.format(word.text, vec_str))
    
    @staticmethod
    def load(path):
        data = joblib.load(path)
        return Embedding(data['dic'], data['trg'], data['ctx'])

