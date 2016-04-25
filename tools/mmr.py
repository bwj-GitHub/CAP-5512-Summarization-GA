"""
Created on Apr 8, 2016

@author: brandon
"""


import pickle

import nltk
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer

from tools.file_tools import read_file_as_string
from tools.tokenization import SentenceSegmenter, Tokenizer
from tools.rouge import rouge2_bin

class MMR:

    def __init__(self, D):
        self.T = Tokenizer()
        self.D = D  # The Meta-document (a list of sentences (strings)
        self.R = []  # Relevance score for each doc.
        self.bows = None  # Bag of Word rep.s for each sentence
        self.m_bow = None  # BOW rep. for the entire Meta document
        self.vectorizer = CountVectorizer(ngram_range=(1,1))
        self.bows = self.vectorizer.fit_transform(self.D)
        self.m_bow = self.bows.sum(0)

        # Tokenize D (TODO: do this before bows?):
        for i in range(len(self.D)):
            self.D[i] = self.T.tokenize(self.D[i])

        # Calculate relevance scores:
        self.bows = self.bows.toarray()
        for i in range(len(self.D)):
            rel = 1 - cosine(self.bows[i], self.m_bow.transpose())
            self.R.append(rel)

    def summarize(self, words=100, l=.5):
        # l is weight for relevance
        S = []  # Summary
        in_S = set()
        n_words = 0
        s_bow = None  # BOW rep for the summary

        # Add sentences until summary is full
        # TODO: Add rules for finding shorter ending sentences?
        while n_words < words and len(in_S) < len(self.D):
            # Calculate redundancy of each (remaining) sentence:
            scores = []
            for i in range(len(self.D)):
                if i in in_S:
                    continue
                if s_bow is None:
                    red = 0
                else:
                    red = 1 - cosine(self.bows[i], s_bow)

                # Calculate MMR score:
                score = l*self.R[i] - (1-l)*red
                scores.append((score, i))
            scores.sort(reverse=True)

            # Select the highest scoring sentence that will fit in S:
            s = None
            for i in range(len(scores)):
                si = scores[i][1]
                s = self.D[si]
                if n_words + len(s) <= words:
                    break
                else:
                    s = None
            if s is not None:
                S.append(s)
            else:
                return S

            in_S.add(i)
            n_words += len(s)
            try:
                s_bow += self.bows[si]  # element-wise addition?
            except TypeError:
                s_bow = np.zeros(shape=self.m_bow.shape)
                s_bow += self.bows[si]
        return S


# Testing:
if __name__ == "__main__":
    SS = SentenceSegmenter()
    D = read_file_as_string("../Input/Metadocs/d30001t.txt")
    D = SS.segment(D)
    M = MMR(D)
    S1 = M.summarize(words=100, l=.5)
    c = 0
    sum1 = []
    for s in S1:
        sum1.extend(s)
        print(" ".join(s))
        c += len(s)
    print(c)
    print("#############")

    # Get ROUGE-2 Score:
    # Get candidate summary bigrams
    with open("../Input/ref_bigrams.p", "rb") as fp:
        HSums = pickle.load(fp)
    ref_bigrams = HSums["d30001"]
    candidate_bigrams = set(nltk.bigrams(sum1))
    print(rouge2_bin(candidate_bigrams, ref_bigrams))

