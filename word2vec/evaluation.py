import numpy as np
from scipy.stats import spearmanr, pearsonr

class WordSim:
    def __init__(self, word1, word2, scores):
        self.word1 = word1
        self.word2 = word2
        self.scores = scores

    def evaluate(self, emb, r='spearman'):
        word1_index = np.array([w.index for w in self.word1], dtype=np.int32)
        word2_index = np.array([w.index for w in self.word2], dtype=np.int32)

        word1_emb = emb.trg_nrm[word1_index]
        word2_emb = emb.trg_nrm[word2_index]

        model_score = np.sum(word1_emb * word1_emb, axis=1)

        if r == 'spearman':
            return spearmanr(model_score, self.scores)[0]
        elif r == 'pearson':
            return pearsonr(model_score, self.scores)[0]

    @staticmethod
    def load(dic, path, sep=' ', lower=True):
        word1 = []
        word2 = []
        scores = []

        with open(path) as f:
            for line in f:
                line = line.strip()

                if lower:
                    line = line.lower()

                if line.startswith('#'):
                    continue

                w1, w2, score = line.split(sep)

                word1.append(dic.text2word(w1))
                word2.append(dic.text2word(w2))
                scores.append(float(score))

        return WordSim(word1, word2, np.array(scores, dtype=np.float32))
