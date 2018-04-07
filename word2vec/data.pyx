from collections import Counter, defaultdict
import joblib
from ctypes import c_float, c_int32, c_uint32, c_uint64
from libc.stdint cimport int32_t, uint32_t
import multiprocessing
import numpy as np
cimport numpy as np
import scipy.stats
import scipy.spatial.distance

def load_embedding(path):
    word2id = {}

    with open(path) as f:
        n_vocab, dim = tuple(map(int, f.readline().strip().split()))

        matrix = np.zeros(shape=(n_vocab, dim), dtype=np.float32)

        for i, line in enumerate(f):
            word, vec_str = line.split(' ', 1)
            word2id[word] = i
            matrix[i] = np.fromstring(vec_str, sep=' ', dtype=np.float32)

    return word2id, matrix

cdef class Word:
    def __init__(self, index, count, text):
        self.index = index
        self.count = count
        self.text = text

    @property
    def is_UNK(self):
        return self.index == 0

    def __repr__(self):
        return "<Word {} />".format(self.text)

cdef class Vocab:
    def __init__(self, word2idx_count):
        self._id2word = {idx: Word(idx, count, word) for word, (idx, count) 
                in word2idx_count.items()}
        self._text2word = {word.text: word for idx, word in self._id2word.items()}

    cpdef Word text2word(self, str text):
        if text in self._text2word:
            return self._text2word[text]
        else:
            return self._id2word[0]

    cpdef Word id2word(self, int idx):
        return self._id2word[idx]

    cpdef list encode(self, list tokens):
        return [self.text2word(token) for token in tokens]

    cpdef list decode(self, list indexes):
        return [self.id2word(idx) for idx in indexes]

    def __len__(self):
        return len(self._id2word)

    def load_wordsim(self, path):
        with open(path) as f:
            f.readline()
            dataset = [line.strip().lower().split('\t') for line in f]
            dataset = [(self.text2word(word1), self.text2word(word2), float(sim))
                            for word1, word2, sim in dataset]

        return dataset

    def __iter__(self):
        for idx, word in self._id2word.items():
            yield word

    cpdef build_neg_table(self, float power, int table_size):
        cdef list items
        cdef float items_pow, cur
        cdef int index, table_index

        neg_table = multiprocessing.RawArray(c_uint32, table_size)
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

        self._neg_table = np.frombuffer(neg_table, dtype=np.uint32)

    cpdef Word sample_negative(self):
        cdef int table_index, word_index
        cdef Word word

        if self._neg_table is None:
            raise Exception('Call build_neg_table first')

        table_index = np.random.randint(0, self._neg_table.shape[0])
        word_index = self._neg_table[table_index]
        word = self.id2word(word_index)
        return word

    @staticmethod
    def build(path):
        word_count = Counter()
        with open(path) as f:
            for line in f:
                tokens = line.split()
                word_count.update(tokens)

        word2idx_count = {word: (idx, count) for idx, (word, count) 
                in enumerate(word_count.items(), 1)}
        word2idx_count['__UNK__'] = (0, 0)

        return Vocab(word2idx_count)

    def save(self, path):
        word2idx_count = {word.text: (word.index, word.count) for word in self}

        joblib.dump(word2idx_count, path)

    @staticmethod
    def load(path):
        word2idx_count = joblib.load(path)
        return Vocab(word2idx_count)

cdef class Dictionary:
    def __init__(self, src_vocab, trg_vocab, word2tran):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.trans = {}

        for src_word, trg_words in word2tran.items():
            src_word = self.src_vocab.text2word(src_word)
            trg_words = [self.trg_vocab.text2word(w) for w in trg_words]
            trg_words = [w for w in trg_words if not w.is_UNK]

            if src_word.is_UNK or not trg_words:
                continue

            self.trans[src_word.index] = trg_words

    cpdef list get_trans(self, Word word):
        if word.index in self.trans:
            return self.trans[word.index]
        else:
            return None

    def build(self, iterator):
        dictionary = defaultdict(list)

        for src_word, trg_word in iterator:
            dictionary[src_word].append(trg_word)

        dictionary = {src_word: list(set(trg_words)) for src_word, trg_words in dictionary}

        return dictionary

cdef class Corpus:
    def __init__(self, vocab, path, n_lines):
        self._vocab = vocab
        self._path = path
        self._n_lines = n_lines
        self._processed_line = 0
        self._epoch = 0

    @property
    def n_lines(self):
        return self._n_lines

    @property
    def processed_line(self):
        return self._processed_line

    @property
    def epoch(self):
        return self._epoch

    @property
    def progress(self):
        return self._processed_line / self._n_lines
    
    def __iter__(self):
        cdef str line
        with open(self._path) as f:
            while True:
                self._epoch += 1
                f.seek(0)
                for line in f:
                    indexes = self._vocab.encode(line.split())
                    yield indexes
                    self._processed_line += 1
                    if self._processed_line >= self._n_lines:
                        return

cdef class Embedding:
    def __init__(self, vocab, trg, ctx):
        self.vocab = vocab
        self.trg = trg
        self.ctx = ctx
        self.trg_nrm = np.linalg.norm(self.trg, axis=1)

    def save(self, path):
        joblib.dump(dict(vocab=self.vocab, trg=self.trg, ctx=self.ctx), path)

    def save_text(self, path):
        with open(path, 'w') as f:
            f.write('{} {}\n'.format(self.trg.shape[0], self.trg.shape[1]))
            for word in self.vocab:
                vec_str = ' '.join('{:.6f}'.format(v) for v in self.get_vec(word))
                f.write('{} {}\n'.format(word.text, vec_str))
    
    @staticmethod
    def load(path):
        data = joblib.load(path)
        return Embedding(data['vocab'], data['trg'], data['ctx'])
    
    cpdef np.ndarray get_vec(self, Word word):
        return self.trg[word.index]

    cpdef list get_similar_by_vec(self, np.ndarray vec):
        scores = np.dot(self.trg, vec) / self.trg_nrm / np.linalg.norm(vec)
        indexes = np.argsort(-scores)

        return [(self.vocab.id2word(idx), scores[idx]) for idx in indexes]

    cpdef list get_similar(self, Word word):
        cdef np.ndarray vec = self.get_vec(word)

        return self.get_similar_by_vec(vec)

    def evaluate(self, wordsim):
        model = []
        gold = []

        for word1, word2, sim in wordsim:
            emb1 = self.trg[word1.index]
            emb2 = self.trg[word2.index]

            score = 1-scipy.spatial.distance.cosine(emb1, emb2)
            model.append(score)
            gold.append(sim)

        spearman = scipy.stats.spearmanr(model, gold)[0]
        pearson = scipy.stats.pearson(model, gold)[0]
        return spearman, pearson
